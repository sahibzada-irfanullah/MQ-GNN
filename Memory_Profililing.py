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
from torch.utils.tensorboard import SummaryWriter
from torch.utils import tensorboard as tb
import pynvml


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
    elif args.samp_type == 'sketch':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = SketchSampler(samp_num_list, graph, HW_row_norm=False)
    elif args.samp_type == 'sketchdebias':
        samp_num_list = np.array(args.n_samp * args.samp_growth_rate ** np.arange(2), dtype=int)
        sampler = SketchSamplerWrs(samp_num_list, graph, HW_row_norm=False)
    elif args.samp_type == 'node':
        samp_num_list = [args.n_samp, args.n_samp]
        sampler = dgl.dataloading.NeighborSampler([args.n_samp, args.n_samp])
    filename = "main_{}_{}_L{}_G{}_{}_{}_{}".format(args.dataset, args.samp_type, args.n_layers, args.n_gpus,
                                                    args.Model, args.batch_size, best_model_idx)

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
        best_val, best_tst = -1, -1
        cnt = 0
        batches_time = []
        epoch_time = []
        batch_time_all = []
        valid_f1_single_iter = []
        valid_loss_single_iter = []
        print('-' * 10)
        # log_dir = 'Results/{}/{}'.format(args.dataset, 'log_dir/S_' + str(args.samp_type) + '_M' + args.Model + '_nG' + str(args.n_gpus) + '_Gid' + str(proc_id))
        log_dir = 'Results/{}'.format('log_dir/' + filename)

        writer = tb.SummaryWriter(log_dir)
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(proc_id)
        for epoch in range(args.n_epochs):
            model.train()
            single_epoch_time = []
            temp_batches_time = []

            train_losses = []
            # with model.join():
            with tqdm.tqdm(train_dataloader) as tq:
                for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                    t1 = time.perf_counter_ns()
                    # feature copy from CPU to GPU takes place here
                    inputs = mfgs[0].srcdata['feat']
                    labels = mfgs[-1].dstdata['label']
                    predictions = model(mfgs, inputs)
                    loss = loss_func(predictions, labels)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    t2 = time.perf_counter_ns()
                    temp_batches_time += [(t2 - t1) // 1000000]
                    single_epoch_time += [(t2 - t1) // 1000000]
                    train_losses += [loss.detach().tolist()]
                    del loss
                    # GPU Metrics: Utilization and Memory Usage
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
                    memory_total = memory_info.total / (1024 ** 2)
                    # Log metrics to TensorBoard
                    writer.add_scalar("ComputeUtilization vs Step", gpu_utilization, step)
                    writer.add_scalar("ComputeUtilization vs Time", gpu_utilization, sum(temp_batches_time))
                    writer.add_scalar("MemoryUsed vs Step", memory_used, step)
                    writer.add_scalar("MemoryUsed vs Time", memory_used, sum(temp_batches_time))
                    writer.add_scalar("MemoryTotal vs Step", memory_total, step)
                    writer.add_scalar("MemoryTotal vs Time", memory_total, sum(temp_batches_time))

            batches_time.extend(temp_batches_time)
            # single_epoch_time = remove_elements(single_epoch_time, remove_n_elements)
            epoch_time += [np.sum(single_epoch_time)]

            model.eval()
            # Evaluate on only the first GPU.
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

                    print(("Epoch: %d (%.4fs) Train Loss: %.2f  Valid Metric: %.3f") % \
                          (epoch, sum(batches_time[-args.batch_num:]), np.average(train_losses), valid_f1))

                    valid_f1_single_iter.append(valid_f1)
                    valid_loss_single_iter.append(loss_valid.item())
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
        pynvml.nvmlShutdown()
        writer.flush()
        writer.close()


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
    parser.add_argument('--batch_size', type=int, default=2560,
                        help='size of output node in a batch')
    # parser.add_argument('--n_iters', type=int, default=1,
    #                     help='Number of iteration to run on a batch')
    parser.add_argument('--n_trial', type=int, default=1,
                        help='Number of times to repeat experiments')
    parser.add_argument('--samp_growth_rate', type=float, default=1,
                        help='Growth rate for layer-wise sampling')
    parser.add_argument('--batch_num', type=int, default=1,
                        help='Maximum Batch Number')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GNN layers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args(dataset_name = 'ogbn-arxiv', samp_temp_name='node', n_samp=5)
    mp.spawn(run, args=(list(range(args.n_gpus)), args,), nprocs=args.n_gpus, join=True)


