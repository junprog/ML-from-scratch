import numpy as np

import nn.Module as nn
from utils import AverageMeter, Logger, IrisDataset

class NeuralNet:
    def __init__(self, in_dim=8, activation='sigmoid'):
        self.fc1  = nn.Linear(4,in_dim)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_dim,3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.fc1(x)
        x = self.activation(x)
        out = self.fc2(x)
        loss = self.criterion(out, target)

        return out, loss

    def backward(self):
        dy = self.criterion.backward()
        dy = self.fc2.backward(dy)
        dy = self.activation.backward(dy)
        dy = self.fc1.backward(dy)

    def update_params(self, lr=1e-1):
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad

if __name__ == '__main__':

    ## dfine logger & meter
    train_logger = Logger('train.log', ['epoch', 'loss', 'acc'])
    test_logger = Logger('test.log', ['epoch', 'loss', 'acc'])

    losses = AverageMeter()
    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv")
    train_data, test_data = iris.split()

    ## Hyper params
    epoch = 300
    lr = 0.01

    ## define Model
    model = NeuralNet(in_dim=8, activation='sigmoid')

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