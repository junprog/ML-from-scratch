from os import X_OK
import numpy as np

import NaiveBayes.NB_classfier as NB
from utils import AverageMeter, IrisDataset

if __name__ == '__main__':

    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv", norm=True)
    train_data, test_data = iris.split(test_set_rate=1/5)

    model = NB.NaiveBayes(mode='gaussian')

    X = train_data[:, 0:4]
    Y = train_data[:, 4:7]

    model.fit(X, Y) ## training the model

    pred = model.pred(test_data[:,0:4]) ## prediction
    
    for i, p in enumerate(pred):
        flag = 0
        print('{} \t {}'.format(p, test_data[i,4:7]))
        if np.argmax(p) == np.argmax(test_data[i,4:7]):
            flag = 1
        accs.update(flag)

    print('Acurracy : ', accs.avg)