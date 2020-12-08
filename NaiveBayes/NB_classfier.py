import numpy as np
from numpy.core.fromnumeric import _sum_dispatcher

class NaiveBayes():
    def __init__(self, mode='gaussian') -> None:
        super().__init__()

        self.mode = mode
        self.sigma = None
    
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

        log_prior = np.log(self.pi)

        sum_list = []
        for d in data:
            sum = []
            sum = [np.sum(self._norm(d, self.mu[c], self.sigma[c]), axis=0) for c in range(self.num_class)]
            sum_list.append(sum)

        log_posterior = np.array(sum_list)
        
        return log_prior + log_posterior

    def _norm(self, x, mu, sigma):
        tmp =  np.exp(-(x - mu)**2 / (2*sigma**2))
        return (1 / (np.sqrt(2*np.pi)*sigma)) * tmp
