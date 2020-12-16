import csv

import numpy as np
import pandas as pd

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.log_file.flush()

class IrisDataset:
    np.random.seed(seed=765) ## set random seed
    def __init__(self, data_path, norm=True):
        ## Data load
        self.data = np.loadtxt(data_path, delimiter=",", skiprows=1, usecols=(0,1,2,3))
        if norm == True:
            self.data = self._normal(self.data)
        self.gt = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols=4, dtype=str)

        ## One hot encoding
        onehot = [s.replace("Setosa", str(0)).strip('"') for s in self.gt]
        onehot = [s.replace("Versicolor", str(1)).strip('"') for s in onehot]
        onehot = [s.replace("Virginica", str(2)).strip('"') for s in onehot]
        onehot = [int(n) for n in onehot]
        self.target = np.identity(3)[onehot]

    def split(self, test_set_rate=1/5):
        ## split train & test set
        train_num = int(self.data.shape[0] * (1 - test_set_rate))

        all_idx = np.random.choice(self.data.shape[0], self.data.shape[0], replace=False)
        train_idx = all_idx[0:train_num]
        test_idx = all_idx[train_num:self.data.shape[0]]

        train_data = {'data' : self.data[train_idx], 'target' : self.target[train_idx]}
        test_data = {'data' : self.data[test_idx], 'target' : self.target[test_idx]}

        return train_data, test_data

    def cross_validate_set(self, k=5):
        ## return k-patterns train-test set (taple)
        assert self.data.shape[0] % k == 0, "[Warning] can not equal split"
        test_num = int(np.round(self.data.shape[0] / k))
        all_idx = np.random.choice(self.data.shape[0], self.data.shape[0], replace=False)

        cross_set = []
        for i in range(k):                
            test_idx = all_idx[i*test_num:(i+1)*test_num]
            test_data = {'data' : self.data[test_idx], 'target' : self.target[test_idx]}
            train_data = {'data' : np.delete(self.data, test_idx, axis=0), 'target' : np.delete(self.target, test_idx, axis=0)}
            cross_set.append((train_data, test_data))

        return cross_set

    def _normal(self, data):
        out = (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0))
        return out

class TitanicDataset:
    def __init__(self,  data_path):
        a = 0