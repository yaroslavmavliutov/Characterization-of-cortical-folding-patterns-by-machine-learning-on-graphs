import os.path as op
import os
import networkx as nx
import numpy as np
import math
import torch
from torch_geometric.data import Data

def graph_reader(path):
    """
    Reading graphs from disk.
    :param path: Path to the file.
    :return data: object of networkx.
    """
    try:
        graphs_files = os.listdir(path)
        return graphs_files
    except:
        print("Could not load data")
        exit()


def array_to_dict(array):
    D = {}
    for i in range(array.shape[0]):
        try:
            D[i] = [array[i][0], array[i][1], array[i][2]]
        except:
            D[i] = array[i][0]
    return D


def from_networkx(G):
    """"Converts a networkx graph to a pytorch's data object graph.
    Args:
        G (networkx.Graph): A networkx graph.
    """
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    keys = []
    keys += list(G.nodes(data=True)[0].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}

    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for key, item in data.items():
        data[key] = torch.tensor(item)
    data['edge_index'] = edge_index

    return Data.from_dict(data)


# convert a list of nx.graph to list of torch.tensor
def load_data(graphs):
    features = [from_networkx(g) for g in graphs]
    return features

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class GCDataset():
    """The dataset class.
    The dataset contains all our graphs.
    """
    def __init__(self):
        super(GCDataset, self).__init__()
        self.num_graphs = 0
        self.graphs = []
        self.subjects = []
        self.labels = []
        self.hem = ['lh']
        self.path = '/home/yaroslav/PycharmProjects/MedicalImageAnalysis/newdata'
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.num_graphs

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Return: networkx obj, numpy array, string
        """
        return self.graphs[idx], self.labels[idx], self.subjects[idx]

    def _getall(self):
        return np.asarray(self.graphs), np.asarray(self.labels)

    def _generate(self):
        self._gen_graph()

    def _gen_graph(self):
        graphs = graph_reader(self.path)
        for graph in graphs:
            for hem in self.hem:
                if hem in graph:
                    graph_path = op.join(self.path, graph)
                    G = nx.read_gpickle(graph_path)
                    self.graphs.append(self._get_labels(G))
                    self.subjects.append(graph)
                    self.num_graphs +=1

    def _get_labels(self, G):
        # definition lables by the depth
        # depth = np.asarray(nx.attr_matrix(G, node_attr='depth')[1])
        # threshold = np.median(np.sort(depth)) < depth
        # self.labels.append((threshold * 1))

        label = []
        for i in range(len(G)):
            if G.nodes[i]['labels'] == 1.0:
                label.append(0.0)
            elif G.nodes[i]['labels'] == 2.0:
                label.append(1.0)
            else:
                label.append(G.nodes[i]['labels'])
        self.labels.append(np.asarray(label))
        return G
