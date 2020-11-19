import numpy as np
import nn.Module as nn

class NeuralNet:
    def __init__(self):
        self.fc1  = nn.Linear(4,8)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(8,3)

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
    model = NeuralNet()

    inputs = np.random.randn(1,4)
    target = np.array([[1,0,0]])


    for i in range(100):
        out, loss = model.forward(inputs, target)
        print(loss)

        model.backward()
        model.update_params()

    print("grad fc2_weight : ", model.fc2.weight_grad.shape)
    print("grad fc2_bias : ", model.fc2.bias_grad.shape)
    print("grad fc1_weight : ", model.fc1.weight_grad.shape)
    print("grad fc1_bias : ", model.fc1.bias_grad.shape)

        