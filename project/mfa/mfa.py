from __future__ import division
import numpy as np
from scipy import linalg
import os
import matplotlib.pyplot as plt
from ..modules.simulate import FactorModel
import pdb

class MixtureFA(object):
    """
        MixtureFA
        m : number of factor models
        k : number of factors for each factor model
        ll : log likelihood curve
        tol : convergence criteria

        n_var : number of variables in the dataset
        lambdas : (m * n_var, k) matrix of factor loadings
        phi : diagonal specificities matrix
        pi : prior probability of each factor model
        mu : the mean vector for each factor model
    """
    def __init__(self, m, k):
        self.n_mixtures = m
        self.n_factors = k
        self.ll = []
        self.tol = 1E-4
        self.iterations = 0

    def setup_params(self, X, seed=5, save_init=False):
        self.sample_size = X.shape[0]
        self.n_var = X.shape[1]
        cov = np.cov(X.T)
        scale = np.power(linalg.det(cov),1/self.n_var)
        np.random.seed(seed)
        self.lambdas = np.random.normal(size = self.n_var * self.n_mixtures * self.n_factors).reshape(self.n_var * self.n_mixtures, self.n_factors) * np.sqrt(scale / self.n_factors)
        self.phi = np.diag(cov)
        self.pi = np.ones(self.n_mixtures) / self.n_mixtures
        self.mu = np.random.normal(size = self.n_mixtures * self.n_var).reshape(self.n_mixtures, self.n_var).dot(linalg.sqrtm(cov)) + np.ones((self.n_mixtures, 1)).dot(np.mean(X, axis=0).reshape(1,self.n_var))
        self.H = np.zeros((self.sample_size, self.n_mixtures))
        if save_init:
            with open(os.path.join('.','initial_value.txt'), 'w') as outfile:
                outfile.write("Initial MU\n")
                outfile.write(np.array2string(self.mu, separator=',') + '\n')
                outfile.write("Initial lambdas\n")
                outfile.write(np.array2string(self.lambdas, separator=',') + '\n')
                outfile.write("Initial phi\n")
                outfile.write(np.array2string(self.phi, separator=',') + '\n')
                outfile.write("Initial pi\n")
                outfile.write(np.array2string(self.pi, separator=',') + '\n')

    def fit(self, X, cyc = 100,seed=5):
        np.random.seed(seed)
        tiny = np.exp(-700)
        self.setup_params(X)
        Ez = np.zeros((self.sample_size * self.n_mixtures, self.n_factors))
        Ezz = np.zeros((self.n_factors * self.n_mixtures, self.n_factors))
        s = np.zeros(self.n_mixtures)

        while True:
            phi_m = np.diag(1/self.phi)
            for j in xrange(self.n_mixtures):
                lambdas = self.lambdas[j*self.n_var:(j+1)*self.n_var, :]
                PL = phi_m.dot(lambdas)
                S = phi_m - PL.dot(linalg.inv( np.eye(self.n_factors) + np.transpose(lambdas).dot(PL) )).dot(PL.T)
                dS = np.sqrt(linalg.det(S))
                Xj = X - np.vstack(np.ones(self.sample_size)).dot(self.mu[j,:].reshape(1, self.n_var))
                XS = Xj.dot(S)
                Ez[j*self.sample_size:(j+1)*self.sample_size, :] = XS.dot(lambdas)
                self.H[:,j] = np.power(2*np.pi, -self.n_var/2)*self.pi[j]*dS*np.exp(-0.5*np.diag(XS.dot(Xj.T)))

            # remove 0 in H
            Hrsum = np.sum(self.H, axis=1)
            zero_idx = np.where(Hrsum == 0)
            n_zero = len(zero_idx)
            lik = np.sum(np.log(Hrsum + n_zero * tiny))

            self.H[zero_idx, :] = tiny * np.ones((n_zero, self.n_mixtures)) / self.n_mixtures
            Hrsum[zero_idx] = tiny * np.ones(n_zero)

            self.H = self.H / np.vstack(Hrsum)
            #compute s
            s = np.sum(self.H, axis=0)
            s2 = np.sum(s)
            # print(s)

            if self.iterations > 0 and ( lik - self.ll[-1] < self.tol ):
                break
            #compute log likelihood
            self.ll.append(lik)
            self.iterations += 1

            for j in xrange(self.n_mixtures):
                lambdas = self.lambdas[j*self.n_var:(j+1)*self.n_var, :]
                PL = phi_m.dot(lambdas)
                S = phi_m - PL.dot(linalg.inv( np.eye(self.n_factors) + np.transpose(lambdas).dot(PL) )).dot(PL.T)
                Xj = X - np.vstack(np.ones(self.sample_size)).dot(self.mu[j,:].reshape(1, self.n_var))
                XX = np.transpose( Xj * np.vstack(self.H[:,j]) ).dot(Xj) / s[j]
                beta = lambdas.T.dot(S)
                Ezz[j*self.n_factors:(j+1)*self.n_factors, :] = np.eye(self.n_factors) - beta.dot(lambdas) + beta.dot(XX).dot(beta.T)

            #compute new params
            ph = np.zeros(self.n_var)
            for j in xrange(self.n_mixtures):
                l0 = np.vstack(self.H[:,j]) * X
                Ez_aug = np.vstack((Ez[j*self.sample_size:(j+1)*self.sample_size, :].T, np.ones(self.sample_size))).T
                l1 = l0.T.dot(Ez_aug)
                XH = Ez[j*self.sample_size:(j+1)*self.sample_size, :].T.dot(self.H[:, j])
                Ezz_aug = np.zeros((self.n_factors+1, self.n_factors+1))
                Ezz_aug[:self.n_factors,:self.n_factors] = s[j] * Ezz[j*self.n_factors:(j+1)*self.n_factors,:]
                Ezz_aug[self.n_factors, :self.n_factors] = XH
                Ezz_aug[:self.n_factors, self.n_factors] = XH
                Ezz_aug[self.n_factors,self.n_factors] = s[j]
                l2 = l1.dot(linalg.inv(Ezz_aug))
                self.lambdas[j*self.n_var:(j+1)*self.n_var, :] = l2[:,:self.n_factors]
                self.mu[j, :] = l2[:, self.n_factors]
                # pdb.set_trace()
                p1 = np.diag(l0.T.dot(X) - l2.dot(l1.T)) / s2
                ph += p1
            self.phi = ph
            self.pi = s / s2


    def plot_lik(self, file_name = None):
        x = np.arange(self.iterations)
        plt.xlabel('Iterations')
        plt.ylabel('LL')
        plt.plot(x, self.ll)
        if file_name:
            plt.savefig(os.path.join('.',"%s.png" % file_name), bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def predict(self, x):
        """predict which mixture the obs comes from by maximizing the posterior prob"""
        prob = np.zeros(self.n_mixtures)
        phi_m = np.diag(1/self.phi)
        for j in xrange(self.n_mixtures):
            lambdas = self.lambdas[j*self.n_var:(j+1)*self.n_var, :]
            PL = phi_m.dot(lambdas)
            S = phi_m - PL.dot(linalg.inv( np.eye(self.n_factors) + np.transpose(lambdas).dot(PL) )).dot(PL.T)
            dS = np.sqrt(linalg.det(S))
            Xj = x.reshape(1, self.n_var) - self.mu[j,:].reshape(1, self.n_var)
            XS = Xj.dot(S)
            prob[j] = np.power(2*np.pi, -self.n_var/2)*self.pi[j]*dS*np.exp(-0.5*np.diag(XS.dot(Xj.T)))
        prob_sum = np.sum(prob)
        scaled_prob = prob / prob_sum
        return self.mixture_index()[np.argmax(scaled_prob)]

    def mixture_index(self):
        """calculate the index for the mixtures sorted by the average value of mu"""
        mixture_with_index = [(i, np.sum(row)) for i, row in enumerate(self.mu)]
        return list(zip(*sorted(mixture_with_index, key=lambda k: k[1]))[0])

    def number_of_parameters(self):
        return np.product(self.lambdas.shape) + np.product(self.mu.shape) + len(self.phi) + len(self.pi)

    def aic(self):
        """2*(number of estimated parameters) - 2*ln(max likelihood)"""
        return 2*self.number_of_parameters() - 2*self.ll[-1]

    def bic(self):
        """ln(sample size) * (number of estimated parameters) - 2*ln(max likelihood)"""
        return np.log(self.sample_size) * self.number_of_parameters() - 2*self.ll[-1]

    def save_estimates(self, file_name):
        with open(os.path.join('.','estimates_%s.txt' % file_name), 'w') as outfile:
            outfile.write("MU\n")
            outfile.write(np.array2string(self.mu, separator=',') + '\n')
            outfile.write("lambdas\n")
            outfile.write(np.array2string(self.lambdas, separator=',') + '\n')
            outfile.write("phi\n")
            outfile.write(np.array2string(self.phi, separator=',') + '\n')
            outfile.write("pi\n")
            outfile.write(np.array2string(self.pi, separator=',') + '\n')

if __name__ == "__main__":
    std = np.sqrt([0.1,0.1,0.1,0.1])
    lambdas = np.array([[1,1,1,1]])
    mu1 = np.array([0,0,0,0])
    mu2 = np.array([10,10,10,10])
    mu3 = np.array([20,20,20,20])

    fm1 = FactorModel(100, lambdas.T, std, mu=mu1, seed=10)
    fm1.simulate()
    fm2 = FactorModel(100, lambdas.T, std, mu=mu2, seed=10)
    fm2.simulate()
    fm3 = FactorModel(100, lambdas.T, std, mu=mu3, seed=10)
    fm3.simulate()
    np.random.seed(5)
    # X1 = np.random.multivariate_normal([0,0,0,0],np.eye(4),100)
    # X2 = np.random.multivariate_normal([2,2,2,2],np.eye(4),200)
    X = np.vstack((fm1.obs, fm2.obs, fm3.obs))
    # print(np.sum(fm3.obs+mu3,axis=0)/300)
    mfa = MixtureFA(2, 1)
    mfa.fit(X)
    # # print(mfa.iterations)
    # print(mfa.H)
    print mfa.mu
    print mfa.lambdas
    print mfa.pi
    print mfa.ll[-1]
    # mfa.plot_lik("3_mix")







