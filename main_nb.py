from os import X_OK
import numpy as np

import NaiveBayes.NB_classfier as NB
from utils import AverageMeter, Logger, IrisDataset

if __name__ == '__main__':

    ## dfine logger & meter
    train_logger = Logger('train.log', ['epoch', 'loss', 'acc'])
    test_logger = Logger('test.log', ['epoch', 'loss', 'acc_0', 'acc_1', 'acc_2'])

    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv", norm=True)
    train_data, test_data = iris.split()

    model = NB.NaiveBayes(mode='gaussian')

    X = train_data[:, 0:4]
    Y = train_data[:, 4:7]

    model.fit(X, Y)

    pred = model.pred(test_data[:,0:4])
    for i, p in enumerate(pred):
        if np.argmax(p) == np.argmax(test_data[i,4:7]):
            print('{} \t {}'.format(p, test_data[i,4:7]))
            accs.update(1)

    print(accs.avg)