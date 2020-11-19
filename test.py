import numpy as np
import nn.Module as nn

fc1 = nn.Linear(4,8)
sig = nn.Sigmoid()
fc2 = nn.Linear(8,3)

print(fc1)
print(sig)
print(fc2)

criterion = nn.CrossEntropyLoss()

print("fc1_weight shape : ", fc1.weight.shape)
print("fc2_weight shape : ", fc2.weight.shape)

inputs = np.random.randn(1,4)
target = np.array([[1,0,0]])

fc1_out = fc1(inputs)
sig_out = sig(fc1_out)
fc2_out = fc2(sig_out)

loss = criterion(fc2_out, target)

def softmax(x):
    out = (x - np.max(x)) 
    return np.exp(out) / np.exp(out).sum()

print("outputs : " , fc2_out)
print("logits : ", softmax(fc2_out))
print("logits sum : ", softmax(fc2_out).sum())
print("target : ", target)
print("CE : ", loss)

grad_loss = criterion.backward()
grad_fc2 = fc2.backward(grad_loss)
grad_sig = sig.backward(grad_fc2)
grad_fc1 = fc1.backward(grad_sig)

print("grad loss : ", grad_loss)
print("grad fc2_weight : ", fc2.weight_grad.shape)
print("grad fc2_bias : ", fc2.bias_grad.shape)
print("grad fc1_weight : ", fc1.weight_grad.shape)
print("grad fc1_bias : ", fc1.bias_grad.shape)