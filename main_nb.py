import numpy as np

import NaiveBayes.NB_classifier as NB
from utils import AverageMeter, IrisDataset

if __name__ == '__main__':

    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv", norm=True)
    train_data, test_data = iris.split(test_set_rate=1/5)

    model = NB.NaiveBayes(mode='gaussian')

    model.fit(train_data['data'], train_data['target']) ## training the model

    pred = model.pred(test_data['data']) ## prediction
    
    ## evalation
    for i, p in enumerate(pred):
        flag = 0
        print('{} \t {}'.format(p, test_data['target'][i]))
        if np.argmax(p) == np.argmax(test_data['target'][i]):
            flag = 1
        accs.update(flag)

    print('Acurracy : ', accs.avg)