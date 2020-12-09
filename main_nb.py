import numpy as np
from numpy.core.numeric import cross

import NaiveBayes.NB_classifier as NB
from utils import AverageMeter, IrisDataset

if __name__ == '__main__':

    print('--- Hold-out validation ---')
    accs = AverageMeter()

    ## data load
    iris = IrisDataset("data/iris.csv", norm=True)
    train_data, test_data = iris.split(test_set_rate=1/5)

    model1 = NB.NaiveBayes(mode='gaussian') ## define the model

    model1.fit(train_data['data'], train_data['target']) ## training the model

    pred = model1.pred(test_data['data']) ## prediction
    
    ## evalation
    for i, p in enumerate(pred):
        flag = 0
        #print('{} \t {}'.format(p, test_data['target'][i]))
        if np.argmax(p) == np.argmax(test_data['target'][i]):
            flag = 1
        accs.update(flag)

    print('Acurracy : ', accs.avg)

    print('\n')

    k_fold = 5
    print('--- {}-fold Cross validation ---'.format(k_fold))
    cross_list = iris.cross_validate_set(k=k_fold)

    model2 = NB.NaiveBayes(mode='gaussian') ## define the model

    acc_list = []
    for train_data, test_data in cross_list:
        accs.reset()
        model2.fit(train_data['data'], train_data['target'])
        pred = model2.pred(test_data['data']) ## prediction
        
        ## evalation
        for i, p in enumerate(pred):
            flag = 0
            #print('{} \t {}'.format(p, test_data['target'][i]))
            if np.argmax(p) == np.argmax(test_data['target'][i]):
                flag = 1
            accs.update(flag)
        
        acc_list.append(accs.avg)

    print('Acurracy : ', acc_list)