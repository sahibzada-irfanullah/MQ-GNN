import os, sys

os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import math
import numpy as np
import torch.nn.functional as F
import torch
import tqdm
import sklearn.metrics
import time
import datetime, argparse, sys
from utils import load_data, record_result_new, remove_elements
import pickle as pkl
from sklearn.metrics import roc_auc_score
from sampler import LayerDependentSampler, FastGCNSampler, FastGCNSamplerCustom, LayerDependentSamplerWrs, \
    SketchSampler, SketchSamplerWrs
from model import Model_GCN, Model_GraphSAGE
import torch.multiprocessing as mp


def run(proc_id, devices, args):
    # Initialize distributed training context.
    graph, node_labels, train_nids, valid_nids, test_nids = load_data(args)
    graph = dgl.add_self_loop(graph)
    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]

    num_classes = (node_labels.max() + 1).item()

    multi_label = True if args.dataset in ['ogbn-proteins'] else False

    if multi_label:
        loss_func = F.binary_cross_entropy_with_logits
    else:
        loss_func = F.cross_entropy
    graph.create_formats_()
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port=args.master_port)
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    best_model_idx = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')
    if args.samp_type == 'ladies':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = LayerDependentSampler(samp_num_list, graph)
    elif args.samp_type == 'fastgcn':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = FastGCNSampler(samp_num_list, graph)
    elif args.samp_type == 'fastgcnflat':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = FastGCNSamplerCustom(samp_num_list, graph, flat=True)
    elif args.samp_type == 'fastgcndebias':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = FastGCNSamplerCustom(samp_num_list, graph, wrs=True)
    elif args.samp_type == 'fastgcnflatdebias':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = FastGCNSamplerCustom(samp_num_list, graph, flat=True, wrs=True)
    elif args.samp_type == 'ladiesdebias':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = LayerDependentSamplerWrs(samp_num_list, graph, HW_row_norm=False)
    elif args.samp_type == 'ladiesflat':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = LayerDependentSampler(samp_num_list, graph, flat=True)
    elif args.samp_type == 'ladiesflatdebias':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = LayerDependentSamplerWrs(samp_num_list, graph, flat=True)
    elif args.samp_type == 'node':
        sampler = dgl.dataloading.NeighborSampler([args.n_samp, args.n_samp])
    filename = "main_{}_{}_L{}_G{}_{}_{}".format(args.dataset, args.samp_type, args.n_layers, args.n_gpus, args.Model,
                                                 best_model_idx)

    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0  # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )



    # create main directory: "Results/args.dataset"
    dir_name = '{}/{}'.format('Results', args.dataset)
    # create directory for saving a best model, i.e., "model": "Results/args.dataset/model"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    dir_name = '{}/{}/{}'.format('Results', args.dataset, 'model')
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for oiter in range(args.n_trial):
        if args.Model == "GCN":
            model = Model_GCN(num_features, args.nhid, num_classes).to(device)
        elif args.Model == "GraphSAGE":
            model = Model_GraphSAGE(num_features, args.nhid, num_classes).to(device)
        # Wrap the model with distributed data parallel module.
        if device == torch.device('cpu'):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

        # Define optimizer
        opt = torch.optim.Adam(model.parameters())

        print('-' * 10)
        log_dir = 'Results/{}'.format('log_dir/' + filename)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                profile_memory=True,
                with_stack=True
        ) as prof:
            for epoch in range(args.n_epochs):
                model.train()
                with tqdm.tqdm(train_dataloader) as tq:
                    for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                        inputs = mfgs[0].srcdata['feat']
                        labels = mfgs[-1].dstdata['label']

                        predictions = model(mfgs, inputs)
                        loss = loss_func(predictions, labels)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        del loss

                model.eval()
                if proc_id == 0:
                    predictions = []
                    labels = []
                    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                        for input_nodes, output_nodes, mfgs in tq:
                            inputs = mfgs[0].srcdata['feat']
                            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                        predictions = np.concatenate(predictions)
                        labels = np.concatenate(labels)
                        # if not multi_label:
                        #     labels = torch.from_numpy(labels).float()
                        #     predictions = torch.from_numpy(predictions).float()
                        # loss_valid = loss_func(predictions, labels)
                        if multi_label:  # 'proteins'
                            loss_valid = loss_func(predictions, labels)
                            valid_f1 = sklearn.metrics.roc_auc_score(labels, predictions)
                        else:  # ['cora', 'citeseer', 'pubmed', 'reddit', 'arxiv', 'products']
                            labels = torch.from_numpy(labels).float()
                            predictions = torch.from_numpy(predictions).float()
                            loss_valid = loss_func(predictions, labels)
                            if args.dataset in ["reddit", "cora", "citeseer",
                                                "pubmed"]:  # ['cora', 'citeseer', 'pubmed', 'reddit']
                                valid_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
                            else:  # 'arxiv', 'products'
                                valid_f1 = sklearn.metrics.accuracy_score(labels, predictions)

                        print(("Epoch: %d (%.4fs)   Valid Metric: %.3f") % (epoch, valid_f1))


                        if valid_f1 > best_val + 1e-2:
                            best_val = valid_f1
                            # "main_{}_fastgcn_{}_{}".format(args.dataset, args.n_layers, best_model_idx)
                            torch.save(model,
                                       '{}/{}/{}/best_model_{}_L{}_G{}_{}_{}.pt'.format('Results', args.dataset, 'model',
                                                                                        args.samp_type, args.n_layers,
                                                                                        args.n_gpus, args.Model,
                                                                                        best_model_idx))
                            cnt = 0
                        else:
                            cnt += 1
                        if cnt == args.n_stops // args.batch_num:
                            break
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        if proc_id == 0:
            best_model = torch.load(
                '{}/{}/{}/best_model_{}_L{}_G{}_{}_{}.pt'.format('Results', args.dataset, 'model', args.samp_type,
                                                                 args.n_layers, args.n_gpus, args.Model,
                                                                 best_model_idx), map_location=device)
            best_model.to(device)
            best_model.eval()
            test_f1s = []
            predictions = []
            labels = []
            with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(best_model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                # accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                if multi_label:  # 'proteins'
                    test_f1 = sklearn.metrics.roc_auc_score(labels, predictions)
                else:  # ['cora', 'citeseer', 'pubmed', 'reddit', 'arxiv', 'products']
                    # labels = torch.from_numpy(labels).float()
                    # predictions = torch.from_numpy(predictions).float()
                    # loss_valid = loss_func(predictions, labels)
                    if args.dataset in ["reddit", "cora", "citeseer",
                                        "pubmed"]:  # ['cora', 'citeseer', 'pubmed', 'reddit']
                        test_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
                    else:  # 'arxiv', 'products'
                        test_f1 = sklearn.metrics.accuracy_score(labels, predictions)
                # test_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
                test_f1s += [test_f1]

            print('Iteration: %d, Test Metric: %.3f' % (oiter, np.average(test_f1s)))



        #             torch.save(model.state_dict(), best_model_path)


def init_args(dataset_name, samp_temp_name, n_samp):
    parser = argparse.ArgumentParser(
        description='Training GCN/GraphSAGE on cora/citeseer/pubmed/proteins/arxiv/reddit/prdoucts Datasets')

    parser.add_argument('--dataset', type=str, default=dataset_name,
                        help='Dataset name: cora/citeseer/pubmed/proteins/arxiv/reddit/products')
    parser.add_argument('--samp_type', type=str, default=samp_temp_name,
                        help='Sampling type: node/fastgcn/fastgcncustom/ladies/ladieswrs/sketchsampler/sketch/sketchwrs')
    # parser.add_argument('--samp_type', type=str, default='ladies', help='Sampling type: node/ladies/fastgcn')

    parser.add_argument('--Model', type=str, default='GraphSAGE',
                        help='Model name: GCN/GraphSAGE')

    parser.add_argument('--n_samp', type=int, default=n_samp,
                        help='Number of sampled nodes per layer or per node')
    parser.add_argument('--nhid', type=int, default=256,
                        help='Hidden state dimension')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of Epoch')
    parser.add_argument('--n_stops', type=int, default=200,
                        help='Stop after number of batches that f1 dont increase')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Specify number of GPUs')
    parser.add_argument('--master_port', type=str, default='12370',
                        help='Specify master port')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='size of output node in a batch')
    # parser.add_argument('--n_iters', type=int, default=1,
    #                     help='Number of iteration to run on a batch')
    parser.add_argument('--n_trial', type=int, default=1,
                        help='Number of times to repeat experiments')
    parser.add_argument('--batch_num', type=int, default=1,
                        help='Maximum Batch Number')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GNN layers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # node/fastgcn/fastgcnflat/fastgcnwrs/ladies/ladieswrs/sketch/sketchwrs
    # samp_type_name = 'node' , 'fastgcn', 'fastgcnflat', 'fastgcnwrs','fastgcnflatwrs' , 'ladies', 'ladiesflat', 'ladieswrs', 'ladiesflatwrs'
    samp_type_list = ['node', 'fastgcn', 'fastgcnflat', 'fastgcnwrs', 'fastgcnflatwrs', 'ladies', 'ladiesflat',
                      'ladieswrs', 'ladiesflatwrs']
    samp_type_list = ['node']
    # samp_type_list = ['fastgcnwrs','fastgcnflatwrs' , 'ladies', 'ladiesflat', 'ladieswrs', 'ladiesflatwrs']
    for samp_type_name in samp_type_list:
        args = init_args(dataset_name='ogbn-arxiv', samp_temp_name='node', n_samp=5)
        mp.spawn(run, args=(list(range(args.n_gpus)), args,), nprocs=args.n_gpus, join=True)


