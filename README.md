# ML-from-scratch
Creating Machine Learning Model from Scratch with Numpy

## To Do
* [x] Neural Nework
    * [ ] Current dataset(concat(tarin,test)) -> dict('train' : trainset, 'test' : testset)
* [ ] Naive Bayes Classfier

## 1. Neural Network

### 1.1 Essestials
* Neural Network model
    * Linear layer
    * Activation function
        * ReLU
        * Sigmoid

* Loss function
    * Cross Entoropy
    
* Optimizer
    * Gradient Descent

### 1.2 How to use
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
    
    def update_params(self, lr=1e-1): ## update params with GD
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad

model = NN() ## instance

## Code training roop 
...

```

### 1.3 Run
* requirements
    * python >= 3.6
    * numpy == 1.18.1 (ambiguous)
    * scipy == 1.5.2  (ambiguous)
    * pandas (only titanic datasets)
```bash
$ python main.py
```