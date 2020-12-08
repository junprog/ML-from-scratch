import re
import numpy as np
from numpy.core.fromnumeric import _sum_dispatcher

class NaiveBayes():
    def __init__(self, mode='gaussian') -> None:
        super().__init__()

        self.mode = mode
        self.sigma = None
        self.eps = 1e-323
    
    def fit(self, data, target): ## [入力 : data (features)]が得られる時に[出力 : target (label)]が得られる[確率 : p(Y|X)]を最大化
        assert data.shape[0] == target.shape[0], "The num of sample of X & Y is different"

        self.num_class = target.shape[1]
        self.num_sample = data.shape[0]
        self.dim_features = data.shape[1]

        self.mu = np.zeros((self.num_class, self.dim_features))
        self.sigma = np.zeros((self.num_class, self.dim_features))
        self.pi = np.zeros(self.num_class)

        if self.mode == 'gaussian':
            num_each_class = np.sum(target, axis=0)
            self.pi = np.true_divide(num_each_class, self.num_sample)

            for l in range(self.num_class):
                sum = np.sum(data[i] if np.argmax(target[i]) == l else 0.0 for i in range(self.num_sample))
                self.mu[l] = sum / num_each_class[l]

            for l in range(self.num_class):
                sum = np.sum((data[i] - self.mu[l])**2 if np.argmax(target[i]) == l else 0.0 for i in range(self.num_sample))
                self.sigma[l] = sum / num_each_class[l]

    def pred(self, data):
        assert self.sigma is not None, "This model have not been fitted yet"

        likeli_list = []
        for d in data:
            log_likeli = np.log(self.pi) + [np.sum(self._log_norm(d, self.mu[c], self.sigma[c]), axis=0) for c in range(self.num_class)]
            likeli_list.append(log_likeli)
        
        return np.array(likeli_list)

    def _log_norm(self, x, mu, sigma):
        term_1 = -(1/2)*np.log(2*np.pi)
        term_2 = -np.log(sigma + self.eps)
        term_3 = -((x - mu)**2) / ((2*sigma) + self.eps)
        return term_1 + term_2 + term_3
