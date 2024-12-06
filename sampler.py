import dgl
from utils import matrix_row_normalize, estWRS_weights, normalize_lap
import scipy.sparse as sp
import numpy as np
import torch
class LayerDependentSampler(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g, flat=True):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = matrix_row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()
        self.flat = flat

    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            prob_i = np.array(np.sum(Q.multiply(Q), axis=0))[0]
            if self.flat: prob_i = np.sqrt(prob_i)
            prob = prob_i / np.sum(prob_i)
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
            next_nodes_list = np.unique(np.concatenate((next_nodes_list, batch_nodes)))
            adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list])
            adj = matrix_row_normalize(adj)
            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = next_nodes_list
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return torch.tensor(prev_nodes_list), batch_nodes, subgs




class FastGCNSampler(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = normalize_lap(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()


    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        prob_i = np.array(np.sum(self.lap_matrix.multiply(self.lap_matrix), axis=0))[0]
        prob = prob_i / np.sum(prob_i)

        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
            next_nodes_list = np.unique(np.concatenate((next_nodes_list, batch_nodes)))
            adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list]/s_num).tocsr()
            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = subgs[-1].srcnodes()
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return prev_nodes_list.clone().detach(), batch_nodes, subgs

class FastGCNSamplerCustom(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g, HW_row_norm = False, flat=False, wrs=False):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = normalize_lap(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()
        self.wrs = wrs
        self.flat = flat


    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        prob_i = np.array(np.sum(self.lap_matrix.multiply(self.lap_matrix), axis=0))[0]
        if self.flat: prob_i = np.sqrt(prob_i)
        prob = prob_i / np.sum(prob_i)

        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            if self.wrs:
                next_nodes_list, weights = estWRS_weights(prob, s_num)
                adj = Q[: , next_nodes_list].multiply(weights).tocsr()
            else:
                next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
                adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list]/s_num).tocsr()
            # next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
            # next_nodes_list = np.unique(np.concatenate((next_nodes_list, batch_nodes)))
            # adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list]/s_num).tocsr()

            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]

            prev_nodes_list = subgs[-1].srcnodes()
        subgs.reverse()

        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return prev_nodes_list.clone().detach(), batch_nodes, subgs


class LayerDependentSamplerWrs(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g, flat = False, HW_row_norm = False):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = matrix_row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()
        self.flat = flat

    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            prob_i = np.array(np.sum(Q.multiply(Q), axis=0))[0]
            if self.flat: prob_i = np.sqrt(prob_i)
            prob = prob_i / np.sum(prob_i)
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list, weights = estWRS_weights(prob, s_num)
            adj = Q[: , next_nodes_list].multiply(weights).tocsr()
            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = next_nodes_list
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return torch.as_tensor(prev_nodes_list).clone().detach(), batch_nodes, subgs


class SketchSampler(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g, HW_row_norm = False):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = matrix_row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()
        self.H_W_row_norm = HW_row_norm

    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            prob_i = np.sqrt(np.array(np.sum(Q.multiply(Q), axis=0))[0])
            if not (self.H_W_row_norm is False):
                prob_i2 = HW_row_norm[self.layers-1-l, : ]
                prob_i  = prob_i * prob_i2
            prob = prob_i / np.sum(prob_i)
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
            adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list]/s_num).tocsr()
            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = next_nodes_list
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return torch.as_tensor(prev_nodes_list).clone().detach(), batch_nodes, subgs


class SketchSamplerWrs(dgl.dataloading.Sampler):
    def __init__(self, fanouts, g, HW_row_norm = False):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = matrix_row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()
        self.HW_row_norm = HW_row_norm

    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            prob_i = np.sqrt(np.array(np.sum(Q.multiply(Q), axis=0))[0])
            if not (self.HW_row_norm is False):
                prob_i2 = self.HW_row_norm[self.layers-1-l, : ]
                prob_i  = prob_i * prob_i2
            prob = prob_i / np.sum(prob_i)
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list, weights = estWRS_weights(prob, s_num)
            adj = Q[: , next_nodes_list].multiply(weights).tocsr()
            subgs += [dgl.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = next_nodes_list
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return torch.as_tensor(prev_nodes_list).clone().detach(), batch_nodes, subgs


