import numpy as np
from utils import *
from scipy import linalg
import os
import matplotlib.pyplot as plt


class MlFactorAnalysis(object):
    def __init__(self, n_factors, sample_size, init_params):
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

    def objective(self, cyy, lambdas, taus):
        M = cyy.dot(woodbury(lambdas, taus))
        return -(np.log(linalg.det(M)) - np.trace(M) + 9)

    def iterative_step(self, cyy, lambdas, taus):
        """
            update params iteratively
        """
        new_lambdas = lambdas.dot(woodbury(lambdas, taus)).dot(cyy)
        new_taus = np.diag( np.diag( cyy - np.transpose(new_lambdas).dot(new_lambdas) ) )
        return new_lambdas, new_taus

    def fit(self, cyy):
        #set initial value of ll, lambdas, taus
        lambdas, taus = self.set_initial(cyy)
        ll = self.objective(cyy, lambdas, taus)
        #while loop until abs(ll* - ll) < delta
        while True:
            new_lambdas, new_taus = self.iterative_step(cyy, lambdas, taus)
            new_ll = self.objective(cyy, new_lambdas, new_taus)
            self.iterations += 1

            print "Num iterations: %d" % (self.iterations)
            print "Likelihood = %f" % (new_ll)
            print

            if self.iterations >=1000 or abs(new_ll - ll) < self.delta:
                return new_ll, new_lambdas, new_taus
            else:
                lambdas = new_lambdas
                taus = new_taus
                ll = new_ll

    def test_iterations(self, cyy, lambdas, taus):
        l , t = self.iterative_step(cyy, lambdas, taus)
        return l, t, self.objective(cyy, l, t)

    def plot_learning_curve(self, cyy):
        lambdas, taus = self.set_initial(cyy)
        ll = self.objective(cyy, lambdas, taus)
        lls = [ll]
        while True:
            new_lambdas, new_taus, new_ll = self.test_iterations(cyy, lambdas, taus)
            self.iterations += 1
            lls.append(new_ll)
            #print "Num iterations: %d" % (self.iterations)
            #print "Likelihood = %f" % (new_ll)
            #print

            if abs(ll - new_ll) < self.delta:
            #if self.iterations == 50:
                break
            else:
                lambdas = new_lambdas
                taus = new_taus
                ll = new_ll
        x = np.arange(0, self.iterations + 1)
        plt.plot(x, np.log(lls), color='b')
        plt.xlabel('Iterations')
        plt.ylabel('log(-f(betas, taus))')
        plt.savefig(os.path.join('.', "explore_ML.png"), bbox_inches="tight")

    def set_initial(self, cyy):
        """
            set initial value of params
        """
        print "Need lambdas to be %d*%d matrix" %(self.num_fators, cyy.shape[0])
        lambdas = self.init_params["lambdas"]
        taus = self.init_params["taus"]
        return lambdas, taus

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

    mlfa = MlFactorAnalysis(4,100, init_params)
    mlfa.plot_learning_curve(cyy)
