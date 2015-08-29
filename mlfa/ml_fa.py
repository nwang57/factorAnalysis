import numpy as np
from scipy import linalg

class MlFactorAnalysis(object):
    def __init__(self, n_factors, sample_size):
        """
            number of variables = p
            n_factors = q
            so lambda is q*p matrix, taus is p*p diag matrix
        """
        self.num_fators = n_factors
        self.iterations = 0
        self.delta = 0.001
        self.n = sample_size

    def objective(self, cyy, lambdas, taus):
        return -0.5 * self.n * ( np.log(linalg.det(taus + np.transpose(lambdas).dot(lambdas))) + np.trace(np.dot(cyy, self.woodbury(lambdas, taus))) )

    def iterative_step(self, cyy, lambdas, taus):
        """
            update params iteratively
        """
        new_lambdas = lambdas.dot(self.woodbury(lambdas, taus)).dot(cyy)
        new_taus = np.diag( np.diag( cyy - np.transpose(new_lambdas).dot(new_lambdas) ) )
        return new_lambdas, new_taus

    def fit(self, cyy):
        #set initial value of ll, lambdas, taus
        lambdas, taus = set_initial(cyy)
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


    def woodbury(self, lambdas, taus):
        """
            woodbury identity util function for inv(taus + transpose(lambdas)*lambdas)
        """
        taus_inv = linalg.inv(taus)
        left = taus_inv.dot(np.transpose(lambdas))
        return taus_inv - left.dot( linalg.inv(np.identity(self.num_fators) + np.dot(lambdas,taus_inv).dot(np.transpose(lambdas))) ).dot(np.transpose(left))

    def set_initial(cyy):
        """
            set initial value of params
        """
        print "Need lambdas to be %d*%d matrix" %(self.num_fators, cyy.shape[0])
        lambdas = np.array([0.7, 0.7])
        taus = np.diag([0.06,0.06])
        return lambdas, taus

if __name__ == "__main__":
    mlfa = MlFactorAnalysis(1,20)
