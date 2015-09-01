from nose.tools import *
import numpy as np
from scipy import linalg
from mlfa.emmlfa import EMMlFactorAnalysis
from mlfa.utils import *


class TestEMMlfa:
    def setUp(self):
        self.cyy = np.array([[1.0   ,0.554 ,0.227 ,0.189 ,0.461 ,0.506 ,0.408, 0.280 ,0.241],
                             [0.554 ,1.0   ,0.296 ,0.219 ,0.479 ,0.530 ,0.425, 0.311 ,0.311],
                             [0.277 ,0.296 ,1.0   ,0.769 ,0.237 ,0.243 ,0.304, 0.718 ,0.730],
                             [0.189 ,0.219 ,0.769 ,1.0   ,0.212 ,0.226 ,0.291, 0.681 ,0.661],
                             [0.461 ,0.479 ,0.237 ,0.212 ,1.0   ,0.520 ,0.514, 0.313 ,0.245],
                             [0.506 ,0.530 ,0.243 ,0.226 ,0.520 ,1.0   ,0.473, 0.348 ,0.290],
                             [0.408 ,0.425 ,0.304 ,0.291 ,0.514 ,0.473 ,1.0  , 0.374 ,0.306],
                             [0.280 ,0.311 ,0.718 ,0.681 ,0.313 ,0.348 ,0.374, 1.0   ,0.672],
                             [0.241 ,0.311 ,0.730 ,0.661 ,0.245 ,0.290 ,0.306, 0.672 ,1.0  ]])
        self.lambdas = np.array([[0.70, 0.74, 0.39, 0.37, 0.65, 0.72, 0.60, 0.51, 0.48],
                                 [-0.12,-0.08,0.81, 0.75,-0.03,-0.05, 0.09, 0.65, 0.67],
                                 [0.15, 0.22, 0.33, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00],
                                 [0.00, 0.00, 0.00, 0.00, 0.37, 0.15, 0.35, 0.02,-0.12]])
        self.taus = np.diag([0.48, 0.41, 0.09, 0.31, 0.44, 0.46, 0.52, 0.32, 0.32])
        self.params_ex = {'lambdas' : self.lambdas, 'taus' : self.taus}
        self.emmlfa_ex = EMMlFactorAnalysis(4,100,self.params_ex)

        pattern = np.array([[1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,1]])
        self.params_cm = { 'lambdas' : self.lambdas, 'taus' : self.taus, 'factor_pattern' : pattern }
        self.emmlfa_cm = EMMlFactorAnalysis(4,100,self.params_cm, fa_type="confirmatory")


    def tearDown(self):
        print "TEARDOWN"

    def test_exploratory(self):
        l, t = self.emmlfa_ex.fit(self.cyy)
        assert(abs(np.sum(self.emmlfa_ex.sig_tilt(l,t)) - np.sum(self.cyy)) < 0.104)

    def test_woodbury(self):
        res = woodbury(self.lambdas, self.taus)
        true = linalg.inv(self.taus + np.transpose(self.lambdas).dot(self.lambdas))
        assert_almost_equal(np.sum(res), np.sum(true), 10)

    def test_confirmatory(self):
        l, t = self.emmlfa_cm.fit(self.cyy)
        #assert(abs(np.sum(self.emmlfa_cm.sig_tilt(l,t)) - np.sum(self.cyy)) < 0.104)
