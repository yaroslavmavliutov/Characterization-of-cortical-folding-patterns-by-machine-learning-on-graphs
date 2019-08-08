import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
from src.layers import SAGEConv

import math
import networkx as nx
import numpy as np


class NetSAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, aggregation='add'):
        super(NetSAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, aggregation)
        self.conv2 = SAGEConv(nhid, nclass, aggregation)

    def forward(self, x, adj):
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, adj))
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


class NetGAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(NetGAT, self).__init__()

        self.conv1 = GATConv(nfeat, nhid, heads=15, dropout=0.05)
        self.conv2 = GATConv(nhid*15, nhid, heads=15, dropout=0.05)
        self.conv3 = GATConv(nhid*15, nclass, heads=1)

    def forward(self, x, adj):
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        #x = F.dropout(x, training=self.training)
        x = self.conv3(x, adj)
        return F.log_softmax(x, dim=1)


class NetDCNN(nn.Module):
    """ A DCNN model for node classification.
    A shallow model.
    (K, X) -> DCNN -> Dense -> Out
    """
    def __init__(self, nhop, nfeat, nclass):

        super(NetDCNN, self).__init__()

        self.hops = nhop
        self.features = nfeat
        self.classes = nclass

        self.setup_weights()
        self.reset_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.FloatTensor(self.hops+1, self.features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_matrix.size(1))
        self.weight_matrix.data.uniform_(-stdv, stdv)

    def A_to_diffusion_kernel(self, graph, k, add_one=True):
        """
        Computes [A**0, A**1, A**2, ..., A**k]

        :param A: 2d numpy array
        :param k: degree of series
        :return: 3d numpy array [A**0, A**1, A**2, ..., A**k]
        """
        assert k >= 0
        A = nx.to_numpy_matrix(graph)
        Apow = [np.identity(A.shape[0])]

        if k > 0:
            d = A.sum(0)

            if add_one:
                Apow.append(A / (d + 1.0))
                for i in range(2, k + 1):
                    Apow.append(np.dot(A / (d + 1.0), Apow[-1]))
            else:
                Apow.append(A / d)
                for i in range(2, k + 1):
                    Apow.append(np.dot(A / d, Apow[-1]))

        return torch.from_numpy(np.transpose(np.asarray(Apow, dtype=np.float32), (1, 0, 2)))

    def batch_matmul(self, A, B):
        """
        @params:
        tensor A's shape is (batch_size, m, n)
        tensor B's shape is (n, k)

        @return:
        tensor C's shape is (batch_size, m, k)
        """
        C = torch.einsum('ijk,kl->ijl', A, B)
        return C

    def forward(self, x, graph):
        kernel = self.A_to_diffusion_kernel(graph, self.hops)

        Apow_dot_X = self.batch_matmul(kernel, x)
        Apow_dot_X_times_W = Apow_dot_X * self.weight_matrix
        Z = torch.tanh(Apow_dot_X_times_W)
        Z = Z.view(Z.size(0), -1)
        fcl = NetSimple(Z.size(1), self.classes)
        output = fcl(Z)
        return output

class NetSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NetSimple, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)
