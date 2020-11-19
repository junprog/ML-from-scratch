import numpy as np

def softmax(x):
    out = (x - np.max(x)) 
    return np.exp(out) / np.exp(out).sum()

class Linear:
    def __init__(self, inputs_size, outputs_size):
        self.inputs_size = inputs_size
        self.outputs_size = outputs_size

        self.weight = np.random.randn(inputs_size, outputs_size)
        self.bias = np.random.randn(outputs_size)
    
    def __call__(self, x):
        self.x = x
        return np.dot(self.x, self.weight) + self.bias
    
    def __repr__(self):
        return "Linear(input_size={}, output_size={})".format(self.inputs_size, self.outputs_size)

    def __str__(self):
        return "Linear(input_size={}, output_size={})".format(self.inputs_size, self.outputs_size)

    def backward(self, grad_out):
        self.weight_grad = np.dot(self.x.T, grad_out)
        self.bias_grad = np.sum(grad_out, axis=0)
        return np.dot(grad_out, self.weight.T)

class Sigmoid:
    def __init__(self):
        self.register_out = None
        self.over = 745.000000
        self.under = -709.782712893
    
    def __call__(self, x):
        # overflow or underflow countermeasure
        x = np.where(x > self.over, self.over, x)
        x = np.where(x < self.under, self.under, x)
        self.register_out = 1 / (1 + np.exp((-1)*x))
        return self.register_out

    def __repr__(self):
        return "Sigmoid()"

    def __str__(self):
        return "Sigmoid()"

    def backward(self, grad_out):
        return grad_out * (1.0 - self.register_out) * self.register_out

class ReLU:
    def __init__(self):
        self.register_out = None
    
    def __call__(self, x):
        self.register_out = np.maximum(0, x)
        return self.register_out

    def __repr__(self):
        return "ReLU()"

    def __str__(self):
        return "ReLU()"

    def backward(self, grad_out):
        return np.where(self.register_out > 0, grad_out, 0)

class CrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.target = None

    def __call__(self, outputs, target):
        self.logits = softmax(outputs)
        self.target = target
        return -(target * np.log(self.logits + 1e-323)).sum()
    
    def __repr__(self):
        return "CrossEntropyLoss()"

    def __str__(self):
        return "CrossEntropyLoss()"

    def backward(self):
        return self.logits - self.target