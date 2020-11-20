import csv
import numpy as np
#import scipy.stats

import nn.Module as nn
from utils import AverageMeter, Logger

class NeuralNet:
    def __init__(self, in_dim=8):
        self.fc1  = nn.Linear(4,in_dim)
        self.sigmoid = nn.ReLU()
        self.fc2 = nn.Linear(in_dim,3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.fc1(x)
        x = self.sigmoid(x)
        out = self.fc2(x)
        loss = self.criterion(out, target)

        return out, loss

    def backward(self):
        dy = self.criterion.backward()
        dy = self.fc2.backward(dy)
        dy = self.sigmoid.backward(dy)
        dy = self.fc1.backward(dy)

    def update_params(self, lr=1e-1):
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad

if __name__ == '__main__':
    train_logger = Logger('results/train.log', ['epoch', 'loss', 'acc'])
    test_logger = Logger('results/test.log', ['epoch', 'loss', 'acc'])

    losses = AverageMeter()
    accs = AverageMeter()

    ## Data load
    data = np.loadtxt("data/iris.csv", delimiter=",", skiprows=1, usecols=(0,1,2,3))
    #data = scipy.stats.zscore(data) ## normalization with scipy

    gt = np.loadtxt("data/iris.csv", delimiter = ",", skiprows=1, usecols=4, dtype=str)

    ## One hot encoding
    gt = [s.replace("Setosa", str(0)).strip('"') for s in gt]
    gt = [s.replace("Versicolor", str(1)).strip('"') for s in gt]
    gt = [s.replace("Virginica", str(2)).strip('"') for s in gt]
    gt = [int(n) for n in gt]
    gt = np.identity(3)[gt]

    ## concat data & target
    all_data = np.concatenate([data, gt], 1)

    ## split train & test set
    train_num = int(all_data.shape[0] * 4/5)
    test_num = int(all_data.shape[0] * 1/5)

    all_idx = np.random.choice(all_data.shape[0], all_data.shape[0], replace=False)
    train_idx = all_idx[0:train_num]
    test_idx = all_idx[train_num:all_data.shape[0]]

    train_data = all_data[train_idx]
    test_data = all_data[test_idx]

    ## Hyper params
    epoch = 300
    lr = 0.01

    ## define Model & Loss
    model = NeuralNet(in_dim=8)

    ## Roop
    for epoch in range(epoch):

        ## Training
        losses.reset()
        accs.reset()
        for data in train_data:
            acc_flag = 0

            inputs = data[np.newaxis, 0:4]
            target = data[np.newaxis, 4:7]

            out, loss = model.forward(inputs, target)

            if np.argmax(out) == np.argmax(target):
                acc_flag = 1

            losses.update(loss)
            accs.update(acc_flag)

            model.backward()
            model.update_params(lr=lr)

        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accs.avg})
            
        if epoch % 10 == 0:
            print("[Train] epoch:{} \tloss:{} \tacc:{}".format(epoch, losses.avg, accs.avg))
        
        ## Test
        losses.reset()
        accs.reset()
        for data in test_data:
            acc_flag = 0

            inputs = data[np.newaxis, 0:4]
            target = data[np.newaxis, 4:7]

            out, loss = model.forward(inputs, target)

            if np.argmax(out) == np.argmax(target):
                acc_flag = 1

            losses.update(loss)
            accs.update(acc_flag)

        test_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accs.avg})

        if epoch % 10 == 0:
            print("[Test] epoch:{} \tloss:{} \tacc:{}".format(epoch, losses.avg, accs.avg))