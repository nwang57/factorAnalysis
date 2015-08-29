from nose.tools import *
import numpy as np
from scipy import linalg
from mlfa.ml_fa import MlFactorAnalysis

class TestMlfa:
    def setUp(self):
        self.mlfa = MlFactorAnalysis(1,20)
        self.cyy = np.array([[1,0.664], [0.664,1]])
        self.lambdas = np.array([[0.7,0.2]])
        self.taus = np.diag([0.06,0.06])


    def tearDown(self):
        print "TEARDOWN"

    def test_objective(self):
        res = self.mlfa.objective(self.cyy, self.lambdas, self.taus)
        assert res < 0

    def test_woodbury(self):
        res = self.mlfa.woodbury(self.lambdas, self.taus)
        true = linalg.inv(self.taus + np.transpose(self.lambdas).dot(self.lambdas))
        assert_almost_equal(np.sum(res), np.sum(true), 10)
