import numpy as np
from scipy import linalg
from mlfa.utils import *

class EMMlFactorAnalysis(object):
    def __init__(self,n_factors, sample_size, init_params, fa_type="explore"):
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
        self.type = fa_type

    def objective(self):
        pass

    def sig_tilt(self, lambdas, taus):
        return taus + np.transpose(lambdas).dot(lambdas)

    def iterative_step(self, cyy, lambdas, taus, factor_pattern):
        K = woodbury(lambdas, taus).dot(np.transpose(lambdas))
        B = cyy.dot(K)
        S = np.transpose(K).dot(B) + np.identity(lambdas.shape[0]) - lambdas.dot(K)
        #need filter for confirmatory FA
        if self.type == "confirmatory":
            new_lambdas = np.zeros(lambdas.shape)
            new_taus_array = np.zeros(len(taus))
            for j, pattern in enumerate(factor_pattern):
                #for each varaible j
                new_S = S[pattern,:][:,pattern]
                new_B = B[j,pattern]
                l = linalg.inv(new_S).dot(np.transpose(new_B))
                #update lambdas according to the pattern
                for l_elem, pos in zip(l, pattern):
                    new_lambdas[:, j][pos] = l_elem

                new_taus_array[j] = np.diag(cyy)[j] - new_B.dot(l)

            new_taus = np.diag(new_taus_array)


        else:
            new_lambdas = linalg.inv(S).dot(np.transpose(B))
            new_taus = np.diag(np.diag(cyy - B.dot(new_lambdas)))

        return new_lambdas, new_taus

    def fit(self, cyy):
        lambdas, taus, factor_pattern = self.set_initial(cyy)
        while True:
            new_lambdas, new_taus = self.iterative_step(cyy, lambdas, taus, factor_pattern)
            self.iterations += 1

            #print "Num iterations: %d" % (self.iterations)
            #print "Likelihood = %f" % (new_ll)
            #print

            if np.mean(abs(np.diag(new_taus - taus))) < self.delta:
            #if self.iterations == 50:
                return new_lambdas, new_taus
            else:
                lambdas = new_lambdas
                taus = new_taus


    def set_initial(self, cyy):
        """
            set initial value of params
        """
        print "Need lambdas to be %d*%d matrix" %(self.num_fators, cyy.shape[0])
        lambdas = self.init_params["lambdas"]
        taus = self.init_params["taus"]
        if self.type == "confirmatory":
            factor_pattern = get_index_from_factor_pattern(self.init_params["factor_pattern"])
        else:
            factor_pattern = None
        return lambdas, taus, factor_pattern



if __name__ == "__main__":
    #read params from file
    init_params = {}
    init_params["lambdas"] = np.array([[0.7, 0.7]])
    init_params["taus"] = np.diag([0.06,0.06])

    mlfa = MlFactorAnalysis(1,20, init_params)
