from nose.tools import *
import numpy as np
from scipy import linalg
from mlfa.simulate import FactorModel

class TestFatorModel:
    def setUp(self):
        self.lambdas = np.array([[2,1],[2,1],[1,-1],[1,-1],[1,-1],[1,-1]])
        self.std = np.array([1,2,1,2,1,2])
        self.fm = FactorModel(1000, self.lambdas, self.std)

    def test_covariance(self):
        self.fm.simulate()
        cyy_est = self.lambdas.dot(np.transpose(self.lambdas)) + np.diag(self.std) ** 2
        assert( abs(np.sum(self.fm.cov()) - np.sum(cyy_est)) < 1.0 )
