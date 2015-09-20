import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import os

def woodbury(lambdas, taus):
    '''
        woodbury identity transformation for inv(taus + transpose(lambdas)*lambdas)
    '''
    q = lambdas.shape[0]
    taus_inv = linalg.inv(taus)
    left = taus_inv.dot(np.transpose(lambdas))
    return taus_inv - left.dot( linalg.inv(np.identity(q) + np.dot(lambdas,taus_inv).dot(np.transpose(lambdas))) ).dot(np.transpose(left))

def get_index_from_factor_pattern(pattern):
    '''
        get the matrix index from the factor pattern which is a matrix with 1s and 0s, q*p
    '''
    return [[i for i, v in enumerate(row) if v ] for row in np.transpose(pattern)]

def objective(cyy, lambdas, taus):
    '''
        return the log likelihood computed by the given lambdas and taus
    '''
    M = cyy.dot(woodbury(lambdas, taus))
    return (np.log(linalg.det(M)) - np.trace(M) + 9)
    #return -(np.log(linalg.det(taus + np.transpose(lambdas).dot(lambdas))) + np.trace(M))

def sig_tilt(lambdas, taus):
    '''
        return the estimated population covariance matrix
    '''
    return taus + np.transpose(lambdas).dot(lambdas)

def cal_stat(data):
    '''
        calculate mean and var for the given list
    '''
    return np.mean(data, axis=0), np.var(data, axis=0)

def load_normal_settings():
    taus = np.sqrt([1,2,3,4,5,6,7,8,9])
    lambdas = np.array([[1.3, 1.0, 1.5, 2.3, 1.8, 0.6, 0.2, 0, 1.1],
                        [0.0, 0.0, 0.0, 1.3, 1.6, 0.9, -0.4, -1.8, 0.2],
                        [2.4, 3.1, 1.2, 1.9, 0.5, 0.2, 2.1, 1.6, 0.8]])
    return lambdas.T, taus

def load_lownoise_settings():
    taus = np.sqrt([1,2,3,0.002,5,6,7,0.0008,9])
    lambdas = np.array([[1.3, 1.0, 1.5, 2.3, 1.8, 0.6, 0.2, 0, 1.1],
                        [0.0, 0.0, 0.0, 1.3, 1.6, 0.9, -0.4, -1.8, 0.2],
                        [2.4, 3.1, 1.2, 1.9, 0.5, 0.2, 2.1, 1.6, 0.8]])
    return lambdas.T, taus

def load_highnoise_settings():
    taus = np.sqrt([1, 2, 300, 4, 5, 600, 7, 8,9])
    lambdas = np.array([[1.3, 1.0, 1.5, 2.3, 1.8, 0.6, 0.2, 0, 1.1],
                        [0.0, 0.0, 0.0, 1.3, 1.6, 0.9, -0.4, -1.8, 0.2],
                        [2.4, 3.1, 1.2, 1.9, 0.5, 0.2, 2.1, 1.6, 0.8]])
    return lambdas.T, taus


def plot_comparison(ll1, ll2, file_name=None):
    if len(ll1) > len(ll2):
        max_len = len(ll1)
        append = np.array( [ ll2[-1] ] * (len(ll1) - len(ll2)) )
        ll2 = np.concatenate([ll2, append])
    else:
        max_len = len(ll2)
        append = np.array( [ ll1[-1] ] * (len(ll2) - len(ll1)) )
        ll1 = np.concatenate([ll1, append])

    marked1 = np.isfinite(ll1)
    marked2 = np.isfinite(ll2)
    x = np.arange(max_len)
    plt.xlabel('Iterations')
    plt.ylabel('LL')
    plt.plot(x[marked1], ll1[marked1], color='b', marker='o', label="Max Lik")
    plt.plot(x[marked2], ll2[marked2], color='r', marker='o', label="EM")
    plt.legend(bbox_to_anchor=(0.95, 0.05), loc=4, borderaxespad=0.)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(os.path.join('.', "%s.png" % file_name), bbox_inches="tight")
    plt.close()

def draw_iter_boxplot(ml_iters, em_iters, file_name=None):
    data = [ml_iters, em_iters]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data)
    ax.set_xticklabels(['ML', 'EM'])

    fig.savefig(os.path.join('.', "%s.png" % file_name), bbox_inches='tight')
    plt.close()
