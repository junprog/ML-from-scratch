# NN-from-scratch
Creating Neural Network from Scratch with Numpy (multi-class classification)

## 1. Essestials
* Neural Network model
    * Linear layer
    * Activation function
        * ReLU
        * Sigmoid

* Loss function
    * Cross Entoropy
    
* Optimizer
    * SGD

## 2. How to use

```python
## import this
import nn.Module as nn

## define Neural Network
class NN:
    def __init__(self): ## dfine layrs & loss
        self.fc = nn.Linear(3, 16)
        ...
        self.criterin = nn.CrossEntopyLoss()

    def forward(self, x, t): ## forward pass
        x = self.fc(x)
        ...
        loss = self.criterin(x, t)
        ...

    def backward(self): ## backward pass
        dy = self.criterion.backward()
        ...
        dy = self.fc.backward(dy)
        ...
    
    def update_params(self, lr=1e-1): ## update params with SGD
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad

model = NN() ## instance

## Code training roop 
...

```

## 3. Run
* requirements
    * python >= 3.6
    * numpy == 1.18.1 (ambiguous)
    * scipy == 1.5.2  (ambiguous)
```bash
$ python main.py
```