from __future__ import division
from mfa import MixtureFA
from collections import defaultdict
import os
import numpy as np
from itertools import permutations, combinations
import pdb

class MFACluster(object):
    """
        To estimate number of clusters using MFA with instability
        obs : n*p data matrix, p variables
        max_k : the maximum k in the search range
        c : how many splitting
        n_factors : number of factors per mixture
        ratio : size of training set, default 30% for validation
    """

    def __init__(self, obs, max_k,n_factors,label, iters=10,ratio=0.35):
        self.obs = obs
        self.n_obs = obs.shape[0]
        self.max_k = max_k
        self.n_iters = iters
        self.n_factors = n_factors
        self.m = int(ratio * self.n_obs)
        self.result_matrix = np.zeros((self.n_iters, self.max_k-1))
        self.label = label

    def fit(self):
        iteration = 0
        permu_list = range(self.n_obs)
        for iteration in xrange(self.n_iters):
            m1_data, m2_data, validation, m1_label, m2_label, val_label = self._split_data(permu_list)
            # paralle model fit?
            for k in xrange(2,self.max_k+1):
                print("%d's iteration with k=%d" % (iteration, k))
                m1 = MixtureFA(k, self.n_factors)
                m1.fit(m1_data)
                m2 = MixtureFA(k, self.n_factors)
                m2.fit(m2_data)
                self.result_matrix[iteration, k-2] = self._mismatch(m1, m2, validation)


    def _label_ratio(self, m1_index, m2_index, validate_index):
        m1_label = np.unique(self.label[m1_index], return_counts=True)
        m2_label = np.unique(self.label[m2_index], return_counts=True)
        val_label = np.unique(self.label[validate_index], return_counts=True)
        return m1_label, m2_label, val_label

    def _split_data(self, permu_list):
        """random split the data into 3 groups, 1st and 2nd for modeling with size m,
        3rd for validation with size n-2m.
        """
        np.random.shuffle(permu_list)
        m1_index = permu_list[:self.m]
        m2_index = permu_list[self.m:2*self.m]
        validate_index = permu_list[2*self.m:self.n_obs]
        m1_data = self.obs[m1_index,:]
        m2_data = self.obs[m2_index,:]
        validation_data = self.obs[validate_index,:]
        m1_label, m2_label, val_label = self._label_ratio(m1_index, m2_index, validate_index)
        return m1_data, m2_data, validation_data, m1_label, m2_label, val_label

    def _mismatch(self, m1, m2, validation):
        """for each split and k, calculate number of disagreement between 2 models"""
        count = 0
        for x_i, y_i in combinations(range(validation.shape[0]), 2):
            x, y = validation[x_i,:], validation[y_i,:]
            if (m1.predict(x) == m1.predict(y)) ^ (m2.predict(x) == m2.predict(y)):
                count += 1
        return count

    def best_k(self, method):
        if method == "voting":
            # if this is a tie of minimum distance, then larger group will be choosed.
            ind, counts = np.unique(self.result_matrix.shape[1]-1-np.argmin(np.fliplr(self.result_matrix), axis=1), return_counts=True)
            return ind[np.argmax(counts)] + 2
        elif method == "averaging":
            return np.argmin(np.mean(self.result_matrix, axis=0)) + 2
        elif method == "std":
            std = np.std(self.result_matrix, axis=0)
            mean = np.mean(self.result_matrix, axis=0)
            adjust_mean = mean - 2*std
            for k in reversed(range(len(std))):
                best = True
                for k_prime in xrange(k):
                    if best and adjust_mean[k] > mean[k_prime]:
                        best = False
                        break
                if best:
                    return k + 2
            return 2
        else:
            raise ValueError("No method, please use voting, averaging and std")


if __name__ == "__main__":
    from mfa_experiment import get_sample
    p = [0.3,0.3,0.4]
    X, Y = get_sample(600, p, 6)
    mfa_cluster = MFACluster(X,5,1,ratio=0.4)
    mfa_cluster.fit()
    print mfa_cluster.result_matrix


