import os
from statsmodels.graphics.tests.test_functional import labels
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
import numpy as np
from model import *
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import sklearn.metrics
import time
import datetime
import torch.multiprocessing as mp
import concurrent.futures
import threading
from queue import Queue
import asyncio
from utils import *
from sampler import *
from contextlib import nullcontext


import argparse

parser = argparse.ArgumentParser(description='Training GNN in Parallel on Multiple GPUs')

parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                    help='Dataset name')
parser.add_argument('--GNN_Model', type=str, default='Custom_GNN_Model',
                    help='GCN/GraphSAGE')
parser.add_argument('--samp_type', type=str, default='node',
                    help='Sampling type: node/fastgcn/fastgcncustom/ladies/ladieswrs/sketchsampler/sketch/sketchwrs')
parser.add_argument('--fanout', type=list, default=[4,4],
                    help='Specificy a list of the number of neighbors that a node in a graph is connected to in a specific layer of a graph neural network (GNN) model')
parser.add_argument('--epoch', type=int, default=4,
                    help='Specify number of epochs')
parser.add_argument('--sync_period', type=int, default=4,
                    help='Give synchronization period')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='Specify number of GPUs')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Specify a batch size')
parser.add_argument('--queue_size', type=int, default=4,
                    help='Specify a buffer size')
parser.add_argument('--log-every', type=int, default=20)
parser.add_argument('--eval-every', type=int, default=9)



args = parser.parse_args()

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




def sample_generator(gpu_queue, condition, train_dataloader, valid_dataloader, model, proc_id, in_degree_all_nodes, args):
    d_stream = torch.cuda.Stream()
    best_metric = 0

    for epoch in range(args.epoch):
        if epoch % args.buffer_rs_every == 0:
            if args.queue_size != 0:
                num_nodes = graph.num_nodes()
                num_sample_nodes = int(args.queue_size * num_nodes)
                prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
                args.buffer = np.random.choice(num_nodes, num_sample_nodes, replace=False,
                                           p=prob)
                with tqdm.tqdm(train_dataloader) as tq:
                    for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                        with torch.cuda.stream(d_stream):
                            with condition:
                                condition.acquire()
                                if gpu_queue.full():
                                    condition.wait()
                                gpu_queue.put([mfgs, mfgs[0].srcdata['feat'], mfgs[-1].dstdata['label'], step])
                                condition.notify()
                                condition.release()
        if proc_id == 0:
            model.eval()
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                if multi_label:  # 'proteins'
                    metric = sklearn.metrics.roc_auc_score(labels, predictions)
                else:  # ['cora', 'citeseer', 'pubmed', 'reddit', 'arxiv', 'products']
                    if args.dataset in ["reddit", "cora", "citeseer",
                                        "pubmed"]:  # ['cora', 'citeseer', 'pubmed', 'reddit']
                        metric = sklearn.metrics.f1_score(labels, predictions, average='micro')
                    else:  # 'arxiv', 'products'
                        metric = sklearn.metrics.accuracy_score(labels, predictions)
                print('Epoch {} Validation Accuracy {}'.format(epoch, metric))
                if best_metric < metric:
                    best_metric = metric
    print(50*"*")
    print("Metric", best_metric*100, "\n")
    with condition:
        condition.acquire()
        gpu_queue.put(None)
        condition.notify()
        condition.release()

async def gradient_generator(model, gradient_buffer, con):
            size = float(torch.distributed.get_world_size())
            con.acquire()
            if gradient_buffer.full():
                con.wait()
            parameters_list = list(model.parameters())
            param_avg = []
            for param in parameters_list:
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param_avg.append(param.grad.data/size)
            gradient_buffer.put(param_avg)
            con.notify()
            con.release()


async def gradient_consumer(model, gradient_buffer, con, opt):
            con.acquire()
            if gradient_buffer.empty():
                con.wait()
            param_avg = gradient_buffer.get()
            con.notify()
            con.release()
            for param, param_garad in zip(model.parameters(), param_avg):
                param.grad.data = param_garad
            opt.step()


def average_gradients(model):
    size = float(torch.distirubed.get_world_size())
    for param in model.parameters():
        torch.distirubed.all_reduce(param.grad.data, op=torch.distirubed.ReduceOp.SUM)
        param.grad.data /= size

def get_gradients(model):
    size = float(torch.distributed.get_world_size())
    return [param.grad.data/size for name, param in model.named_parameters()]

async def gradient_generation_consumption(g_stream,model, gradient_buffer, con, opt):
    with torch.cuda.stream(g_stream):
        grad_gen_task = asyncio.create_task(gradient_generator(model, gradient_buffer, con))
    grad_cons_task = asyncio.create_task(gradient_consumer(model, gradient_buffer, con, opt))
    await grad_gen_task
    await grad_cons_task

def sample_consumer(gpu_queue, condition, opt, model, args, BUFFER_SIZE = 4):
    con = threading.Condition()
    gradient_buffer = Queue(maxsize= BUFFER_SIZE)
    c_stream = torch.cuda.Stream()
    m_context = model.no_sync
    iteration = 0
    step = 0
    # m_context = nullcontext
    with torch.cuda.stream(c_stream):
        g_stream = torch.cuda.Stream()
        model.train()
        while True:
            with condition:
                condition.acquire()
                if gpu_queue.empty():
                    condition.wait()
                input_mfg_feat_label = gpu_queue.get()
                condition.notify()
                condition.release()

            if input_mfg_feat_label == None:
                break
            with m_context():
                opt.zero_grad()
                predictions = model(input_mfg_feat_label[0], input_mfg_feat_label[1])
                loss = F.cross_entropy(predictions, input_mfg_feat_label[2])
                loss.backward()
                asyncio.run(gradient_generation_consumption(g_stream,model, gradient_buffer, con, opt))
                labels = input_mfg_feat_label[2].cpu().numpy()
                predictions = predictions.argmax(1).detach().cpu().numpy()
                if step % args.log_every == 0:
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

                if multi_label:  # 'proteins'
                    metric = sklearn.metrics.roc_auc_score(labels, predictions)
                else:  # ['cora', 'citeseer', 'pubmed', 'reddit', 'arxiv', 'products']
                    if args.dataset in ["reddit", "cora", "citeseer",
                                        "pubmed"]:  # ['cora', 'citeseer', 'pubmed', 'reddit']
                        metric = sklearn.metrics.f1_score(labels, predictions, average='micro')
                    else:  # 'arxiv', 'products'
                        metric = sklearn.metrics.accuracy_score(labels, predictions)
                print(f'Metric', metric)
                iteration += 1
                if iteration % args.sync_period == 0:
                    torch.distributed.barrier()
                    average_gradients(model)
                    iteration = 0



def run(proc_id, devices, args):

    # print("GPU Stats in the beginning:", list(map(lambda x: x//divisor, torch.cuda.mem_get_info())) )
    BUFFER_SIZE = args.queue_size
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
    dev_id = devices[proc_id]
    # Initialize distributed training context
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)

    model = globals().get(args.GNN_Model)(num_features, 128, num_classes).to(device)
    number_of_nodes = graph.number_of_nodes()
    in_degree_all_nodes = graph.in_degrees()

    # Compute the node sampling probability.
    prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
    prob_gpu = torch.tensor(prob).to(device)
    avd = int(sum(in_degree_all_nodes)//number_of_nodes)
    # Define model and optimizer
    in_degree_all_nodes = in_degree_all_nodes.to(device)
    # Define training and validation dataloader
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor layer_dependent_sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=True,       # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,    # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of layer_dependent_sampler processes
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

    # Wrap the model with distributed data parallel module.
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Define optimizer
    opt = torch.optim.Adam(model.parameters())


    condition = threading.Condition()
    gpu_queue = Queue(maxsize=BUFFER_SIZE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(sample_generator, gpu_queue, condition, train_dataloader, valid_dataloader, model, proc_id, in_degree_all_nodes, args)
        executor.submit(sample_consumer, gpu_queue, condition, opt, model,args, BUFFER_SIZE)


graph.create_formats_()

if __name__ == '__main__':
    # num_gpus = args.num_gpus
    mp.spawn(run, args=(list(range(args.num_gpus)), args,), nprocs=args.num_gpus)
