import numpy as np
from scipy import linalg as la

class PCA(object):
    def __init__(self, n_comp):
        self.n_comp = n_comp

    def fit(self, cyy):
        #using covariance matrix to perform the pca
        evals, evecs = la.eigh(cyy)
        idx = np.argsort(evals)[::-1]
        self.evecs = evecs[:,idx]
        self.evals = evals[idx]
        self.selected_evecs = self.evecs[:, :self.n_comp]
        self.selected_evals = self.evals[:self.n_comp]
        return self.selected_evals, self.selected_evecs


