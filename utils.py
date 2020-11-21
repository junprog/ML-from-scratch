import csv

import numpy as np
import scipy.stats
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
    np.random.seed(seed=765) ## random seed
    def __init__(self, data_path, norm=True):
        ## Data load
        self.data = np.loadtxt(data_path, delimiter=",", skiprows=1, usecols=(0,1,2,3))
        if norm == True:
            self.data = scipy.stats.zscore(self.data) ## normalization with scipy
        self.gt = np.loadtxt(data_path, delimiter = ",", skiprows=1, usecols=4, dtype=str)

        ## One hot encoding
        onehot = [s.replace("Setosa", str(0)).strip('"') for s in self.gt]
        onehot = [s.replace("Versicolor", str(1)).strip('"') for s in onehot]
        onehot = [s.replace("Virginica", str(2)).strip('"') for s in onehot]
        onehot = [int(n) for n in onehot]
        self.onehot = np.identity(3)[onehot]

        ## concat data & target
        self.all_data = np.concatenate([self.data, self.onehot], 1)

    def split(self, test_set_rate=1/5):
        ## split train & test set
        train_num = int(self.all_data.shape[0] * (1 - test_set_rate))
        test_num = int(self.all_data.shape[0] * test_set_rate)

        all_idx = np.random.choice(self.all_data.shape[0], self.all_data.shape[0], replace=False)
        train_idx = all_idx[0:train_num]
        test_idx = all_idx[train_num:self.all_data.shape[0]]

        self.train_data = self.all_data[train_idx]
        self.test_data = self.all_data[test_idx]

        return self.train_data, self.test_data

class TitanicDataset:
    def __init__(self,  data_path):
        a = 0