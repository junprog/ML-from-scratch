import numpy as np

import nn.Module as nn
from utils import AverageMeter, Logger, IrisDataset

class DeepNeuralNet:
    def __init__(self, in_dim=8, activation='sigmoid'):
        self.fc1  = nn.Linear(4, in_dim)

        if activation == 'sigmoid':
            self.activation1 = nn.Sigmoid()
        elif activation == 'relu':
            self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(in_dim, in_dim*2)

        if activation == 'sigmoid':
            self.activation2 = nn.Sigmoid()
        elif activation == 'relu':
            self.activation2 = nn.ReLU()

        self.fc3 = nn.Linear(in_dim*2, in_dim)

        if activation == 'sigmoid':
            self.activation3 = nn.Sigmoid()
        elif activation == 'relu':
            self.activation3 = nn.ReLU()

        self.fc4 = nn.Linear(in_dim, 3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        x = self.activation3(x)
        out = self.fc4(x)

        loss = self.criterion(out, target)

        return out, loss

    def backward(self):
        dy = self.criterion.backward()
        dy = self.fc4.backward(dy)
        dy = self.activation3.backward(dy)
        dy = self.fc3.backward(dy)
        dy = self.activation2.backward(dy)
        dy = self.fc2.backward(dy)
        dy = self.activation1.backward(dy)
        dy = self.fc1.backward(dy)

    def update_params(self, lr=1e-1):
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * self.fc3.weight_grad
        self.fc3.bias -= lr * self.fc3.bias_grad
        self.fc4.weight -= lr * self.fc4.weight_grad
        self.fc4.bias -= lr * self.fc4.bias_grad
        

if __name__ == '__main__':

    ## dfine logger & meter
    train_logger = Logger('train.log', ['epoch', 'loss', 'acc'])
    test_logger = Logger('test.log', ['epoch', 'loss', 'acc_0', 'acc_1', 'acc_2'])

    losses = AverageMeter()
    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv", norm=True)
    train_data, test_data = iris.split()

    ## Hyper params
    epoch = 301
    lr = 0.01

    ## define Model
    model = DeepNeuralNet(in_dim=8, activation='sigmoid')

    ## Roop
    for epoch in range(epoch):

        ## Training
        losses.reset()
        accs.reset()
        for inputs, target in zip(train_data['data'], train_data['target']):
            acc_flag = 0

            inputs = inputs[np.newaxis,:] ## unsqueeze for transpose
            target = target[np.newaxis,:] ## unsqueeze for transpose

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

    corr_0 = 0.0
    corr_1 = 0.0
    corr_2 = 0.0
    num_0 = 0.0
    num_1 = 0.0
    num_2 = 0.0
    for inputs, target in zip(test_data['data'], test_data['target']):
        acc_flag = 0

        inputs = inputs[np.newaxis, :]
        target = target[np.newaxis, :]

        out, loss = model.forward(inputs, target)

        if np.argmax(out) == np.argmax(target):
            acc_flag = 1
        
        if np.argmax(target) == 0:
            # Setosa
            num_0 += 1.0
            if np.argmax(out) == np.argmax(target):
                corr_0 += 1.0

        if np.argmax(target) == 1:
            # Versicolor
            num_1 += 1.0
            if np.argmax(out) == np.argmax(target):
                corr_1 += 1.0

        if np.argmax(target) == 2:
            # Virginica
            num_2 += 1.0
            if np.argmax(out) == np.argmax(target):
                corr_2 += 1.0


        losses.update(loss)
        accs.update(acc_flag)

    print("Setosa acc : ", corr_0/num_0)
    print("Versicolor acc : ", corr_1/num_1)
    print("Virginica acc : ", corr_2/num_2)

    test_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc_0': corr_0/num_0,
        'acc_1': corr_1/num_1,
        'acc_2': corr_2/num_2,})

    if epoch % 10 == 0:
        print("[Test] epoch:{} \tloss:{} \tacc:{}".format(epoch, losses.avg, accs.avg))