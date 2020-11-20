import numpy as np
import matplotlib.pyplot as plt

data01 = np.loadtxt("test.log", skiprows=1, unpack=True)

fig = plt.figure(figsize=(10, 6))

## Accuracy
ax = fig.add_subplot(1,2,1)
ax.plot(data01[0], data01[2], linestyle="-", color="tab:blue", label="Setosa acc")
ax.set_xlim([0,300])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(loc="lower right")

az = fig.add_subplot(1,2,1)
az.plot(data01[0], data01[3], linestyle="-", color="tab:orange", label="Versicolor acc")
az.set_xlim([0,300])
az.set_xlabel("Epoch")
az.set_ylabel("Accuracy")
az.legend(loc="lower right")

ai = fig.add_subplot(1,2,1)
ai.plot(data01[0], data01[4], linestyle="-", color="tab:green", label="Virginica acc")
ai.set_xlim([0,300])
ai.set_xlabel("Epoch")
ai.set_ylabel("Accuracy")
ai.legend(loc="lower right")


### Loss

by = fig.add_subplot(1,2,2)
by.plot(data01[0], data01[1], linestyle="-", color="tab:blue", label="test loss")
by.set_xlim([0,200])
by.set_xlabel("Epoch")
by.set_ylabel("Loss")
by.legend(loc="upper right")

plt.savefig('test_species.svg', format="svg", dpi=1200)
plt.show()