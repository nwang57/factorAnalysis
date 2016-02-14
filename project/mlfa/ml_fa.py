import numpy as np
from scipy import linalg
import os
import matplotlib.pyplot as plt
import pdb
from ..utils import *


class MlFactorAnalysis(object):
    def __init__(self,n_factors, sample_size, init_params, fa_type="explore", plot=True):
        """
            number of variables = p
            n_factors = q
            so lambda is q*p matrix, taus is p*p diag matrix
        """
        self.num_fators = n_factors
        self.iterations = 0
        self.delta = 1E-6
        self.n = sample_size
        self.init_params = init_params
        self.type = fa_type
        self.set_initial()
        self.plot = plot
        self.ll = []

    def set_initial(self):
        """
            set initial value of params
        """
        if self.num_fators != self.init_params["lambdas"].shape[0]:
            raise Exception("number of factors does not match the dimension of loadings")

        self.lambdas = self.init_params["lambdas"]
        self.taus = self.init_params["taus"]
        if self.type == "confirmatory":
            self.factor_pattern = get_index_from_factor_pattern(self.init_params["factor_pattern"])
        else:
            self.factor_pattern = None

    def iterative_step(self, cyy):
        """
            update params iteratively
        """
        K  = self.lambdas.dot(woodbury(self.lambdas, self.taus)).dot(cyy)
        if self.type == "confirmatory":
            new_lambdas = np.zeros(self.lambdas.shape)
            new_taus_array = np.zeros(len(self.taus))
            for j, pattern in enumerate(self.factor_pattern):
                for l_elem, pos in zip(K[pattern,j], pattern):
                    new_lambdas[:, j][pos] = l_elem

                new_taus_array[j] = np.diag(cyy)[j] - np.transpose(K[pattern,j]).dot(K[pattern,j])

            new_taus = np.diag(new_taus_array)
        else:
            new_lambdas = K
            new_taus = np.diag( np.diag( cyy - np.transpose(new_lambdas).dot(new_lambdas) ) )
        self.lambdas, self.taus = new_lambdas, new_taus

    def test_iterations(self, cyy):
        self.iterative_step(cyy)
        return objective(cyy, self.lambdas, self.taus)

    def fit(self, cyy):
        ll = objective(cyy, self.lambdas, self.taus)
        lls = [ll]
        while True:
            new_ll = self.test_iterations(cyy)
            self.iterations += 1
            lls.append(new_ll)
            #print "Num iterations: %d" % (self.iterations)
            #print "Likelihood = %f" % (new_ll)
            #print

            if abs(ll - new_ll) < self.delta:
            #if self.iterations == 50:
                break
            else:
                ll = new_ll

        if self.plot:
            x = np.arange(0, self.iterations + 1)
            plt.plot(x, lls, color='b')
            plt.xlabel('Iterations')
            plt.ylabel('log(-f(betas, taus))')
            if self.type == "confirmatory":
                plt.savefig(os.path.join('.', "confirmatory_ML.png"), bbox_inches="tight")
            else:
                plt.savefig(os.path.join('.', "explore_ML.png"), bbox_inches="tight")

        self.ll = np.array(lls)


if __name__ == "__main__":
    #read params from file
    cyy = np.array([[1.0   ,0.554 ,0.227 ,0.189 ,0.461 ,0.506 ,0.408, 0.280 ,0.241],
                    [0.554 ,1.0   ,0.296 ,0.219 ,0.479 ,0.530 ,0.425, 0.311 ,0.311],
                    [0.227 ,0.296 ,1.0   ,0.769 ,0.237 ,0.243 ,0.304, 0.718 ,0.730],
                    [0.189 ,0.219 ,0.769 ,1.0   ,0.212 ,0.226 ,0.291, 0.681 ,0.661],
                    [0.461 ,0.479 ,0.237 ,0.212 ,1.0   ,0.520 ,0.514, 0.313 ,0.245],
                    [0.506 ,0.530 ,0.243 ,0.226 ,0.520 ,1.0   ,0.473, 0.348 ,0.290],
                    [0.408 ,0.425 ,0.304 ,0.291 ,0.514 ,0.473 ,1.0  , 0.374 ,0.306],
                    [0.280 ,0.311 ,0.718 ,0.681 ,0.313 ,0.348 ,0.374, 1.0   ,0.672],
                    [0.241 ,0.311 ,0.730 ,0.661 ,0.245 ,0.290 ,0.306, 0.672 ,1.0  ]])
    init_params = {}
    init_params["lambdas"] = np.array([[0.70, 0.74, 0.39, 0.37, 0.65, 0.72, 0.60, 0.51, 0.48],
                                       [-0.12,-0.08,0.81, 0.75,-0.03,-0.05, 0.09, 0.65, 0.67],
                                       [0.15, 0.22, 0.33, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00],
                                       [0.00, 0.00, 0.00, 0.00, 0.37, 0.15, 0.35, 0.02,-0.12]])

    init_params["taus"] = np.diag([0.48, 0.41, 0.09, 0.31, 0.44, 0.46, 0.52, 0.32, 0.32])
    init_params["factor_pattern"] = np.array([[1,1,1,1,1,1,1,1,1],
                                              [1,1,1,1,1,1,1,1,1],
                                              [1,1,1,1,0,0,0,0,0],
                                              [0,0,0,0,1,1,1,1,1]])

    mlfa = MlFactorAnalysis(4,100, init_params)
    mlfa.fit(cyy)

    #confir = MlFactorAnalysis(4,100, init_params, fa_type = "confirmatory")
    #confir.fit(cyy)
