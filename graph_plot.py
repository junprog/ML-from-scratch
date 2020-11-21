import numpy as np
import matplotlib.pyplot as plt

"""
name01 = "results/train_unit_2.log"
name02 = "results/test_unit_2.log"

name03 = "results/train_unit_4.log"
name04 = "results/test_unit_4.log"

name05 = "results/train_unit_8.log"
name06 = "results/test_unit_8.log"

name07 = "results/train_unit_16.log"
name08 = "results/test_unit_16.log"
"""

data01 = np.loadtxt("results/train_with_sigmoid.log", skiprows=1, unpack=True)
data02 = np.loadtxt("results/test_with_sigmoid.log", skiprows=1, unpack=True)

data03 = np.loadtxt("results/train_with_sigmoid_without_norm.log", skiprows=1, unpack=True)
data04 = np.loadtxt("results/test_with_sigmoid_without_norm.log", skiprows=1, unpack=True)

data05 = np.loadtxt("results/train_with_ReLU.log", skiprows=1, unpack=True)
data06 = np.loadtxt("results/test_with_ReLU.log", skiprows=1, unpack=True)

data07 = np.loadtxt("results/train_with_ReLU_without_norm.log", skiprows=1, unpack=True)
data08 = np.loadtxt("results/test_with_ReLU_without_norm.log", skiprows=1, unpack=True)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1,2,1)
ax.plot(data01[0], data01[2], linestyle="dashed", color="tab:blue", label="train w/ Sigmoid w/ Norm")
ax.set_xlim([0,300])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(loc="lower right")

az = fig.add_subplot(1,2,1)
az.plot(data03[0], data03[2], linestyle="dashed", color="tab:orange", label="train w/ Sigmoid w/o Norm")
az.set_xlim([0,300])
az.set_xlabel("Epoch")
az.set_ylabel("Accuracy")
az.legend(loc="lower right")

ai = fig.add_subplot(1,2,1)
ai.plot(data05[0], data05[2], linestyle="dashed", color="tab:green", label="train w/ ReLU w/ Norm")
ai.set_xlim([0,300])
ai.set_xlabel("Epoch")
ai.set_ylabel("Accuracy")
ai.legend(loc="lower right")

ai = fig.add_subplot(1,2,1)
ai.plot(data07[0], data07[2], linestyle="dashed", color="tab:red", label="train w/ ReLU w/o Norm")
ai.set_xlim([0,300])
ai.set_xlabel("Epoch")
ai.set_ylabel("Accuracy")
ai.legend(loc="lower right")

#test
ay = fig.add_subplot(1,2,1)
ay.plot(data02[0], data02[2], "-", color="tab:blue", label="test w/ Sigmoid w/ Norm")
ay.set_xlim([0,300])
ay.set_xlabel("Epoch")
ay.set_ylabel("Accuracy")
ay.legend(loc="lower right")

at = fig.add_subplot(1,2,1)
at.plot(data04[0], data04[2], "-", color="tab:orange", label="test w/ Sigmoid w/o Norm")
at.set_xlim([0,300])
at.set_xlabel("Epoch")
at.set_ylabel("Accuracy")
at.legend(loc="lower right")

ai2 = fig.add_subplot(1,2,1)
ai2.plot(data06[0], data06[2], "-", color="tab:green", label="test w/ ReLU w/ Norm")
ai2.set_xlim([0,300])
ai2.set_xlabel("Epoch")
ai2.set_ylabel("Accuracy")
ai2.legend(loc="lower right")

aw2 = fig.add_subplot(1,2,1)
aw2.plot(data08[0], data08[2], "-", color="tab:red", label="test w/ ReLU w/o Norm")
aw2.set_xlim([0,300])
aw2.set_xlabel("Epoch")
aw2.set_ylabel("Accuracy")
aw2.legend(loc="lower right")


### Loss Base line

by = fig.add_subplot(1,2,2)
by.plot(data01[0], data01[1], linestyle="dashed", color="tab:blue", label="train w/ Sigmoid w/ Norm")
by.set_xlim([0,200])
by.set_xlabel("Epoch")
by.set_ylabel("Loss")
by.legend(loc="upper right")

at = fig.add_subplot(1,2,2)
at.plot(data03[0], data03[1], linestyle="dashed", color="tab:orange", label="train w/ Sigmoid w/o Norm")
at.set_xlim([0,200])
at.set_xlabel("Epoch")
at.set_ylabel("Loss")
at.legend(loc="upper right")

ai2 = fig.add_subplot(1,2,2)
ai2.plot(data05[0], data05[1], linestyle="dashed", color="tab:green", label="train w/ ReLU w/ Norm")
ai2.set_xlim([0,200])
ai2.set_xlabel("Epoch")
ai2.set_ylabel("Loss")
ai2.legend(loc="upper right")

aw2 = fig.add_subplot(1,2,2)
aw2.plot(data07[0], data07[1], linestyle="dashed", color="tab:red", label="train w/ ReLU w/o Norm")
aw2.set_xlim([0,200])
aw2.set_xlabel("Epoch")
aw2.set_ylabel("Loss")
aw2.legend(loc="upper right")

by = fig.add_subplot(1,2,2)
by.plot(data02[0], data02[1], linestyle="-", color="tab:blue", label="test w/ sigmoid w/ Norm")
by.set_xlim([0,200])
by.set_xlabel("Epoch")
by.set_ylabel("Loss")
by.legend(loc="upper right")

at = fig.add_subplot(1,2,2)
at.plot(data04[0], data04[1], linestyle="-", color="tab:orange", label="test w/ Sigmoid w/o Norm")
at.set_xlim([0,200])
at.set_xlabel("Epoch")
at.set_ylabel("Loss")
at.legend(loc="upper right")

ai2 = fig.add_subplot(1,2,2)
ai2.plot(data06[0], data06[1], linestyle="-", color="tab:green", label="test w/ ReLU w/ Norm")
ai2.set_xlim([0,200])
ai2.set_xlabel("Epoch")
ai2.set_ylabel("Loss")
ai2.legend(loc="upper right")

aw2 = fig.add_subplot(1,2,2)
aw2.plot(data07[0], data08[1], linestyle="-", color="tab:red", label="test w/ ReLU w/o Norm")
aw2.set_xlim([0,200])
aw2.set_xlabel("Epoch")
aw2.set_ylabel("Loss")
aw2.legend(loc="upper right")

plt.savefig('train_curve.svg', format="svg", dpi=1200)
plt.show()