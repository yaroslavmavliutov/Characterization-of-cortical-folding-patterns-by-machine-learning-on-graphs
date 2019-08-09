import numpy as np
import networkx as nx
import sys

from src.utils import GCDataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import KFold


def main():
    args = sys.argv[1:]
    #args = ['trivial']
    # Load data
    dataset = GCDataset()

    if args[0] == 'stat':
        trivial_statistics(dataset)
    elif args[0] == 'plot':
        # plot graph on 3d
        graph, labels, subject = dataset[2]
        plot_graph_3d(graph)
    elif args[0] == 'dist':
        compute_distribution(dataset, approach='depth')
    elif args[0] == 'trivial':
        X, y = dataset._getall()

        # model estimation by K-Fold
        kf = KFold(n_splits=4)
        score = []
        k = 1
        for train_index, test_index in kf.split(X, y):
            print(k, "-FOLD")
            k += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            threshold = train(X_train)
            scr = test(X_test, y_test, np.mean(threshold))
            score.append(np.mean(scr))
        print(np.mean(score))

def train(Xtrain):
    threshold = []
    for idx, graph in enumerate(Xtrain):
        threshold.append(np.median(np.sort(np.asarray(nx.attr_matrix(graph, node_attr='depth')[1]))))
    return threshold

def test(X_test, y_test, threshold):
    score = []
    for idx, graph in enumerate(X_test):
        depth = np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])
        output = threshold < depth
        score.append(accuracy(output*1, y_test[idx]))
    return score

def accuracy(y_pred, y):
    correct = np.equal(y, y_pred)
    correct = np.sum(correct)
    return correct / len(y)


def plot_graph_3d(graph, show=True):
    points = [v for v in nx.get_node_attributes(graph, 'coord').values()]
    points = np.asanyarray(points, dtype=np.float64)
    if points.shape[1] == 3:
        ax = plt.gca(projection='3d')
        ax.scatter(*points.T, c='r')
        A = nx.to_numpy_matrix(graph)
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if A[i, j] == 1:
                    ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], [points[i][2], points[j][2]], 'b')
                    ax.set_title('3D Graph', loc='left')
                    ax.set_xlabel(xlabel='X-coordinate')
                    ax.set_ylabel(ylabel='Y-coordinate')
                    ax.set_zlabel(zlabel='Z-coordinate')
    if show:
        plt.show()

def compute_distribution(dataset, approach='depth', show=True):
    data, labels = dataset._getall()

    if approach == 'depth':

        total_depth = []
        for graph in data:
            depth = np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])
            for element in depth:
                total_depth.append(element)

        depth = np.asarray(nx.attr_matrix(data[5], node_attr='depth')[1])
        threshold = np.median(np.sort(depth))
        mask = depth <= threshold
        for element in depth:
            total_depth.append(element)
        deep = np.asarray(nx.attr_matrix(data[5], node_attr='depth')[1])[np.nonzero(mask)]
        shallow = np.asarray(nx.attr_matrix(data[5], node_attr='depth')[1])[np.nonzero(np.invert(mask))]
        bins = np.linspace(min(min(deep), min(shallow)), max(max(deep), max(shallow)))
        plt.hist(deep, bins, alpha=0.5, label='Depth of Primary class', color='green')
        plt.hist(shallow, bins, alpha=0.5, label='Depth of Secondary class', color='blue')
        plt.legend(loc='upper right')
        plt.xlabel('Depth')
        plt.ylabel('Nodes')

    elif approach == 'coordinates':

        for idx, graph in enumerate(data):
            mask = labels[idx] == 1.0
            deep = np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])[np.nonzero(mask)]
            shallow = np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])[np.nonzero(np.invert(mask))]
            bins = np.linspace(min(min(deep), min(shallow)), max(max(deep), max(shallow)))
            plt.hist(deep, bins, alpha=0.5, label='Depth of Primary class', color='green')
            plt.hist(shallow, bins, alpha=0.5, label='Depth of Secondary class', color='blue')
            plt.legend(loc='upper right')
            plt.xlabel('Depth')
            plt.ylabel('Nodes')
            plt.savefig('/home/yaroslav/PycharmProjects/MedicalImageAnalysis/plots/plot'+str(idx)+'.png')
            plt.close()

    if show:
        plt.show()


def trivial_statistics(dataset, show = True):
    data, labels = dataset._getall()
    total_nodes_number = []
    total_neighbors_number = []
    total_depth = []
    total_mean = []
    for idx, graph in enumerate(data):
        total_nodes_number.append(nx.number_of_nodes(graph))
        A = nx.to_numpy_matrix(graph)
        for i in range(len(A)):
            total_neighbors_number.append(np.count_nonzero(A[i])-1)

        depth = np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])
        for i in range(len(depth)):
            total_depth.append(depth[i])

        threshold = np.median(np.sort(np.asarray(nx.attr_matrix(graph, node_attr='depth')[1])))
        total_mean.append(threshold)

    count_labels_of_primary_classe = 0
    for i in range(len(labels)):
        count_labels_of_primary_classe = count_labels_of_primary_classe + np.count_nonzero(labels[i])

    fig1, axes1 = plt.subplots(1, 2)
    axes1[0].hist(total_nodes_number)
    axes1[0].set_title('Distribution of total number of nodes')
    axes1[0].set_xlabel(xlabel='Number of nodes')
    axes1[0].set_ylabel(ylabel='Subjects')
    axes1[1].hist(total_neighbors_number)
    axes1[1].set_title('Distribution of total number of neighbors')
    axes1[1].set_xlabel(xlabel='Number of neighbors')
    axes1[1].set_ylabel(ylabel='Nodes')

    fig2, axes2 = plt.subplots()
    axes2.hist(total_mean)
    axes2.set_title('Distribution of thresholds')
    axes2.set_xlabel(xlabel='Depth')
    axes2.set_ylabel(ylabel='Nodes')

    if show:
        plt.show()

if __name__ == '__main__':
    main()