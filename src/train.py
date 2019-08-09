import torch.nn.functional as F
import torch
from tqdm import trange

from src.utils import load_data, GCDataset
from src.models import NetSAGE, NetGAT, NetDCNN
from sklearn.model_selection import KFold

import numpy as np
import copy
from operator import itemgetter


def main():
    dataset = GCDataset()
    X, y = dataset._getall()

    """
    func(X, y, model, features)
    Paramters
    ---------
    X : numpy array
        Features of graphs
    y : numpy array
        Nodes labels of graphs
    model_name : string
        'SAGE', 'GAT' or 'DCNN'
    features :  string
        'adj', 'adj + depth', 'adj + coord' or 'all'
    """
    # for searching optimal hyperparameters
    # which take time (grid search), it gives the following values that are used in classification function
    # best_hyperparameters(X, y, model_name='SAGE', features='all')
    """
    for SAGE model:
    max_epoch = 50
    learning_rate = 0.01
    nhidden = 16
    
    for GAT model:
    max_epoch = 30
    learning_rate = 0.01
    nhidden = 16
    
    for DCNN model:
    max_epoch = 30
    learning_rate = 0.05
    nhidden = 10
    """

    # binary classification
    classification(X, y, model_name='SAGE', features='all')


def classification(X, y, model_name='SAGE', features='all'):

    # checking entered parameters
    assert model_name in ['SAGE', 'GAT', 'DCNN']
    assert features in ['adj', 'adj + depth', 'adj + coord', 'all']

    # Different parameters for each model
    if model_name == 'SAGE':
        max_epoch = 50
        learning_rate = 0.01
        nhidden = 16
    elif model_name == 'GAT':
        max_epoch = 30
        learning_rate = 0.01
        nhidden = 16
    else:
        max_epoch = 30
        learning_rate = 0.05
        nhidden = 10

    # dictionary of feature values to know the size
    features_dict = {
        'adj': 1,
        'adj + depth': 1,
        'adj + coord': 3,
        'all': 4
    }
    features_number = features_dict[features]
    classes_number = 2

    if model_name == 'SAGE':
        # we can choose the aggregation function like 'add', 'mean' or 'max'
        model = NetSAGE(nfeat=features_number, nhid=nhidden, nclass=classes_number, aggregation='add')
    elif model_name == 'GAT':
        model = NetGAT(nfeat=features_number, nhid=nhidden, nclass=classes_number)
    else:
        model = NetDCNN(nhop=nhidden, nfeat=features_number, nclass=classes_number)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    # to save the initial value of the model
    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    kf = KFold(n_splits=6)
    score = []
    k = 1
    for train_index, test_index in kf.split(X, y):
        print(k, "-FOLD")
        k += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # for each kfold iteration we can use the same initial value
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)

        epochs = trange(max_epoch, leave=True, desc="Epoch")
        for epoch in epochs:
            loss, accr = train(model, model_name, optimizer, X_train, y_train, features)
            epochs.set_description("Epoch (Loss=%g  Acc=%g)" % (round(loss, 5), round(accr, 5)))
            scheduler.step()
        test_score = test(model, model_name, X_test, y_test, features)
        print("Test accuracy: ", np.sum(test_score) / len(test_score))
        score.append(np.sum(test_score) / len(test_score))

    print("TOTAL TEST ACCURACY: ", np.sum(score) / len(score))

def train(model, model_name, optimizer, X_train, ytrain, feat):

    model.train()

    # from list of networkx's objects to list of torch.tensors
    xtrain = load_data(X_train)

    total_loss = 0
    total_accuracy = []

    # Stochastic Gradient Descent (batch_size=1)
    for i in range(len(xtrain)):
        optimizer.zero_grad()
        fictive_variable = torch.ones(len(ytrain[i])).resize_((len(ytrain[i])), 1)

        if model_name == 'SAGE' or model_name == 'GAT':
            # feature is only adjacency matrix
            if feat == 'adj': out = model(fictive_variable, xtrain[i].edge_index)
            # features are adjacency matrix and depth
            elif feat == 'adj + depth': out = model(xtrain[i].depth.view(-1, 1), xtrain[i].edge_index)
            # features are adjacency matrix and coordinates
            elif feat == 'adj + coord': out = model(xtrain[i].coord, xtrain[i].edge_index)
            # features are adjacency matrix, depth and coordinates
            else: out = model(torch.cat((xtrain[i].coord, xtrain[i].depth.view(-1, 1)), 1), xtrain[i].edge_index)
        else:
            # for DCNN
            # feature is only adjacency matrix
            if feat == 'adj': out = model(fictive_variable, X_train[i])
            # features are adjacency matrix and depth
            elif feat == 'adj + depth': out = model(xtrain[i].depth.view(-1, 1), X_train[i])
            # features are adjacency matrix and coordinates
            elif feat == 'adj + coord': out = model(xtrain[i].coord, X_train[i])
            # features are adjacency matrix, depth and coordinates
            else: out = model(torch.cat((xtrain[i].coord, xtrain[i].depth.view(-1, 1)), 1), X_train[i])

        y = torch.LongTensor(ytrain[i])
        acc = accuracy(out, ytrain[i])
        total_accuracy.append(acc)
        loss = F.nll_loss(out, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss/len(xtrain), np.mean(total_accuracy)

def test(model, model_name, X_train, ytest, feat):
    model.eval()

    xtest = load_data(X_train)
    total_accuracy = []

    for i in range(len(xtest)):
        fictive_variable = torch.ones(len(ytest[i])).resize_((len(ytest[i])), 1)

        if model_name == 'SAGE' or model_name == 'GAT':
            # adj
            if feat == 'adj': output = model(fictive_variable, xtest[i].edge_index)
            # adj + depth
            elif feat == 'adj + depth': output = model(xtest[i].depth.view(-1, 1), xtest[i].edge_index)
            # adj + coordinates
            elif feat == 'adj + coord': output = model(xtest[i].coord, xtest[i].edge_index)
            # adj + depth + coordinates
            else: output = model(torch.cat((xtest[i].coord, xtest[i].depth.view(-1, 1)), 1), xtest[i].edge_index)
        else:
            # for DCNN
            # adj
            if feat == 'adj': output = model(fictive_variable, X_train[i])
            # adj + dept
            elif feat == 'adj + depth': output = model(xtest[i].depth.view(-1, 1), X_train[i])
            # adj + coordinates
            elif feat == 'adj + coord': output = model(xtest[i].coord, X_train[i])
            # adj + depth + coordinates
            else: output = model(torch.cat((xtest[i].coord, xtest[i].depth.view(-1, 1)), 1), X_train[i])

        acc = accuracy(output, ytest[i])
        total_accuracy.append(acc)
    return total_accuracy

def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = np.equal(preds.detach().numpy(), np.asarray(labels))
    correct = correct.sum()
    return correct / len(labels)


def best_hyperparameters(X, y, model_name='SAGE', features='all'):

    features = 'all'
    features_dict = {
        'adj': 1,
        'adj + depth': 1,
        'adj + coord': 3,
        'all': 4
    }
    features_number = features_dict[features]
    classes_number = 2
    assert model_name in ['SAGE', 'GAT', 'DCNN']
    iteration = 1
    # hyperparameters optimization by grid layout
    for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for max_epoch in [30, 50, 100, 150]:
            for nhop in [10, 16, 24]:
                if model_name == 'SAGE':
                    model = NetSAGE(nfeat=features_number, nhid=nhop, nclass=classes_number, aggregation='add') # 'add', 'mean', 'max'
                elif model_name == 'GAT':
                    model = NetGAT(nfeat=features_number, nhid=nhop, nclass=classes_number)
                else:
                    model = NetDCNN(nhop=nhop, nfeat=features_number, nclass=classes_number)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
                init_state = copy.deepcopy(model.state_dict())
                init_state_opt = copy.deepcopy(optimizer.state_dict())
                kf = KFold(n_splits=6)
                score = []
                result = []
                for train_index, test_index in kf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model.load_state_dict(init_state)
                    optimizer.load_state_dict(init_state_opt)
                    for epoch in range(max_epoch):
                        loss, accr = train(model, model_name, optimizer, X_train, y_train, features)
                        scheduler.step()
                    test_score = test(model, model_name, X_test, y_test, features)
                    score.append(np.sum(test_score) / len(test_score))
                result.append([np.sum(score) / len(score), learning_rate, max_epoch, nhop])
                print(iteration, ' iteration finished')
                iteration +=1
    print(sorted(result, key=itemgetter(0)))
    # We can also define the number of layers for each model individually in src.models
    # But optimal is SAGE(2), GAT(3)

if __name__ == '__main__':
    main()