import numpy as np
from scipy import linalg

class EMMlFactorAnalysis(object):
    def __init__(self,n_factors, sample_size, init_params):
        """
            number of variables = p
            n_factors = q
            so lambda is q*p matrix, taus is p*p diag matrix
        """
        self.num_fators = n_factors
        self.iterations = 0
        self.delta = 0.001
        self.n = sample_size
        self.init_params = init_params

    def objective(self):
        pass

    def iterative_step(self, cyy, lambdas, taus):
        pass

    def fit(self, cyy):
        pass

    def set_initial(self, cyy):
        """
            set initial value of params
        """
        print "Need lambdas to be %d*%d matrix" %(self.num_fators, cyy.shape[0])
        lambdas = self.lambdas
        taus = self.taus
        return lambdas, taus

    def woodbury(self, lambdas, taus):
        """
            woodbury identity util function for inv(taus + transpose(lambdas)*lambdas)
        """
        taus_inv = linalg.inv(taus)
        left = taus_inv.dot(np.transpose(lambdas))
        return taus_inv - left.dot( linalg.inv(np.identity(self.num_fators) + np.dot(lambdas,taus_inv).dot(np.transpose(lambdas))) ).dot(np.transpose(left))
