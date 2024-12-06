import torch.nn as nn
from dgl.nn import SAGEConv
from dgl.nn import GraphConv
import torch.nn.functional as F

class Model_GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model_GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='lstm')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='lstm')
        self.h_feats = h_feats
    #
    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h
#
class Model_GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h


######################################################################