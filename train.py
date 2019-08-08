import torch.nn.functional as F
import torch
from tqdm import trange

from src.utils import load_data, GCDataset
from src.models import NetSAGE, NetGAT, NetDCNN
from sklearn.model_selection import KFold

import numpy as np
import copy


def main():
    dataset = GCDataset()
    X, y = dataset._getall()

    max_epoch = 50
    learning_rate = 0.01
    features_number = 3
    classes_number = 2
    nhidden = 16


    #model = NetSAGE(nfeat=features_number, nhid=nhidden, nclass=classes_number, aggregation='add') # 'add', 'mean', 'max'
    model = NetDCNN(nhop=10, nfeat=features_number, nclass=classes_number)
    #model = NetGAT(nfeat=features_number, nhid=nhidden, nclass=classes_number)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    init_state = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    kf = KFold(n_splits=6)
    score = []
    k = 1
    for train_index, test_index in kf.split(X, y):
        print(k, "-FOLD")
        k+=1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)

        epochs = trange(max_epoch, leave=True, desc="Epoch")
        for epoch in epochs:
            loss, accr = train(model, optimizer, X_train, y_train)
            epochs.set_description("Epoch (Loss=%g  Acc=%g)" % (round(loss, 5), round(accr, 5)))
            scheduler.step()
        test_score = test(model, X_test, y_test)
        print("Test accuracy: ", np.sum(test_score)/len(test_score))
        score.append(np.sum(test_score)/len(test_score))

    print("TOTAL TEST ACCURACY: ", np.sum(score)/len(score))

def train(model, optimizer, X_train, ytrain):

    model.train()

    # from list of networkx to list of torch.tensor
    xtrain = load_data(X_train)

    total_loss = 0
    total_accuracy = []

    # Stochastic Gradient Descent (batch_size=1)
    for i in range(len(xtrain)):
        optimizer.zero_grad()
        fictive_variable = torch.ones(len(ytrain[i])).resize_((len(ytrain[i])), 1)

        # for GAT or SAGE
        # adj
        #out = model(fictive_variable, xtrain[i].edge_index)
        # adj + depth
        #out = model(xtrain[i].depth.view(-1, 1), xtrain[i].edge_index)
        # adj + coordinates
        #out = model(xtrain[i].coord, xtrain[i].edge_index)
        # adj + depth + coordinates
        #out = model(torch.cat((xtrain[i].coord, xtrain[i].depth.view(-1, 1)), 1), xtrain[i].edge_index)

        # for DCNN
        # adj
        #out = model(fictive_variable, X_train[i])
        # adj + depth
        #out = model(xtrain[i].depth.view(-1, 1), X_train[i])
        # adj + coordinates
        out = model(xtrain[i].coord, X_train[i])
        # adj + depth + coordinates
        #out = model(torch.cat((xtrain[i].coord, xtrain[i].depth.view(-1, 1)), 1), X_train[i])

        y = torch.LongTensor(ytrain[i])
        acc = accuracy(out, ytrain[i])
        total_accuracy.append(acc)
        loss = F.nll_loss(out, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss/len(xtrain), np.mean(total_accuracy)

def test(model, X_train, ytest):
    model.eval()

    xtest = load_data(X_train)
    total_accuracy = []

    for i in range(len(xtest)):
        fictive_variable = torch.ones(len(ytest[i])).resize_((len(ytest[i])), 1)

        # for GAT and SAGE
        # adj
        #output = model(fictive_variable, xtest[i].edge_index)
        # adj + depth
        #output = model(xtest[i].depth.view(-1, 1), xtest[i].edge_index)
        # adj + coordinates
        #output = model(xtest[i].coord, xtest[i].edge_index)
        # adj + depth + coordinates
        #output = model(torch.cat((xtest[i].coord, xtest[i].depth.view(-1, 1)), 1), xtest[i].edge_index)

        # for DCNN
        # adj
        #output = model(fictive_variable, X_train[i])
        # adj + dept
        #output = model(xtest[i].depth.view(-1, 1), X_train[i])
        # adj + coordinates
        output = model(xtest[i].coord, X_train[i])
        # adj + depth + coordinates
        #output = model(torch.cat((xtest[i].coord, xtest[i].depth.view(-1, 1)), 1), X_train[i])

        acc = accuracy(output, ytest[i])
        total_accuracy.append(acc)
    return total_accuracy

def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = np.equal(preds.detach().numpy(), np.asarray(labels))
    correct = correct.sum()
    return correct / len(labels)


if __name__ == '__main__':
    main()