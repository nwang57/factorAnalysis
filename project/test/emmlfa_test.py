from nose.tools import *
import numpy as np
from scipy import linalg
from mlfa.emmlfa import EMMlFactorAnalysis
from ..utils import *


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
        self.emmlfa_ex.fit(self.cyy)
        l = self.emmlfa_ex.lambdas
        t = self.emmlfa_ex.taus
        assert(abs(np.sum(sig_tilt(l,t)) - np.sum(self.cyy)) < 1)

    def test_woodbury(self):
        res = woodbury(self.lambdas, self.taus)
        true = linalg.inv(self.taus + np.transpose(self.lambdas).dot(self.lambdas))
        assert_almost_equal(np.sum(res), np.sum(true), 10)

    def test_confirmatory(self):
        self.emmlfa_cm.fit(self.cyy)
        l = self.emmlfa_cm.lambdas
        t = self.emmlfa_cm.taus
        assert(abs(np.sum(sig_tilt(l,t)) - np.sum(self.cyy)) < 1)

    def test_varimax(self):
        test = np.array([[-0.21855 ,  0.908532, -0.32892 ,  0.136455],
                      [ 0.841874, -0.36596 , -0.2355  , -0.31916 ],
                      [ 0.830432,  0.383946, -0.2225  , -0.33684 ],
                      [ 0.381244,  0.169304, -0.20553 ,  0.885294],
                      [ 0.768423,  0.058657,  0.601702,  0.209855],
                      [ 0.361017,  0.036609,  0.928671,  0.076788],
                      [ 0.919046,  0.269187, -0.21876 , -0.18718 ],
                      [ 0.971137,  0.049841,  0.163809,  0.166061],
                      [ 0.334276, -0.82175 , -0.38167 ,  0.25945 ]])
        expect = np.array([[ 0.09129682,  0.88933586, -0.34325038,  0.28797331],
                        [ 0.80891817, -0.58100963,  0.05820452, -0.06850804],
                        [ 0.98631709,  0.14231445,  0.08083308,  0.01976065],
                        [ 0.10435783, -0.00513476,  0.0910481 ,  0.99034953],
                        [ 0.40020067, -0.10117614,  0.87719403,  0.24521912],
                        [-0.00140417,  0.02477792,  0.99722012, -0.07025541],
                        [ 0.97714768,  0.00143377,  0.13723808,  0.1623131 ],
                        [ 0.72024453, -0.2037375 ,  0.55082862,  0.36922558],
                        [ 0.10306332, -0.92631601, -0.20016431,  0.30207721]])
        result = varimax(test)
        assert(abs(np.sum(result) - np.sum(expect)) < 0.1)

