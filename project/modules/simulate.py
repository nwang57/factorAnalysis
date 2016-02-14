import numpy as np
from utils import *
from pca import PCA

class FactorModel(object):
    def __init__(self, n_obs,lambdas, std, mu=None, seed = 5):
        '''
            n_obs: number of observations of y
            lambdas: factor loading, p*q matrix
            std: std of residual, p dim vactor
            seed: set random state for replication
        '''
        self.n_obs = n_obs
        self.lambdas = lambdas
        self.std = std
        self.seed = seed
        self.obs = []
        # if mu is not specified, then assume to be 0
        if None is mu:
            mu = np.zeros(len(self.std))
        self.mu = mu


    def simulate(self):
        '''
            generate n*p matrix of y, n obs, each with p varialbes
        '''
        np.random.seed(self.seed)
        p = len(self.std)
        q = self.lambdas.shape[1]
        #generate n*q hidden fators with mean 0, variance 1
        #size = self.n_obs * q
        fa = np.random.normal(size = self.n_obs * q).reshape(self.n_obs,q)
        stds = np.random.normal(size = self.n_obs * p).reshape(self.n_obs, p).dot(np.diag(self.std))
        y = np.vstack(np.ones(self.n_obs)).dot(self.mu.reshape(1, p)) + fa.dot(np.transpose(self.lambdas)) + stds
        self.obs = y

    def cov(self):
        '''
            return the covariance matrix of the observed y
        '''
        y = self.obs - np.mean(self.obs, axis=0)
        return (np.transpose(y).dot(y) / (self.n_obs-1))

    def get_pca_factors(self, n_factors):
        cyy = self.cov()
        pca = PCA(n_factors)
        evals, evecs = pca.fit(cyy)
        lambdas = np.sqrt(evals) * evecs
        taus = np.diag(np.diag(cyy - lambdas.dot(lambdas.T)))
        return lambdas, taus








if __name__ == "__main__":
    lambdas, std = load_normal_settings()
    fm = FactorModel(1000, lambdas, std)
    fm.simulate()
    y = fm.obs
    cyy = np.transpose(y).dot(y) / (1000-1)

    cyy_est = lambdas.dot(np.transpose(lambdas)) + np.diag(std) ** 2
    print(cyy)

    print(cyy_est)


