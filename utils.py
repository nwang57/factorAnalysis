import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import os

def generate_random_seed(size):
    np.random.seed(-1)
    seeds = list(set(np.random.random_integers(0, 1E5, size=600)))
    if len(seeds) >= 500:
        return seeds[:500]

def woodbury(lambdas, taus):
    '''
        woodbury identity transformation for inv(taus + transpose(lambdas)*lambdas)
    '''
    q = lambdas.shape[0]
    taus_inv = linalg.inv(taus)
    left = taus_inv.dot(np.transpose(lambdas))
    return taus_inv - left.dot( linalg.inv(np.identity(q) + np.dot(lambdas,taus_inv).dot(np.transpose(lambdas))) ).dot(np.transpose(left))

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6, normalize = True):
    """
        varimax implementation from https://en.wikipedia.org/wiki/Talk%3aVarimax_rotation
        normalize each row
        Phi: p*q matrix with each column as a factor
    """
    from scipy import eye, asarray, dot, sum
    if normalize:
        sc = [np.sqrt(sum(row)) for row in Phi**2]
        Phi = Phi / np.vstack(sc)
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = linalg.svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, np.diag(np.diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    result = dot(Phi, R)
    if normalize:
        result = result * np.vstack(sc)
    return result




def sort_factor_by_variance(lambdas):
    """
        lambdas: q*p matrix
    """
    return np.asarray(sorted(lambdas, key = lambda k: sum(k**2))[::-1])

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
    taus = np.sqrt([1,1.2,1.1,0.9,0.8,1,1.3,1.12,0.9])
    lambdas = np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 0, 0, 1]])
    return lambdas.T, taus

def load_lownoise_settings():
    taus = np.sqrt([1,0.012,1.1,0.9,0.8,0.01,1.3,1.12,0.9])
    lambdas = np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 0, 0, 1]])
    return lambdas.T, taus

def load_highnoise_settings():
    taus = np.sqrt([1,120,1.1,0.9,0.8,100,1.3,1.12,0.9])
    lambdas = np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 0, 0, 1]])
    return lambdas.T, taus

def load_stability_test_settings():
    taus = np.sqrt([1,1,1,1,1,1,1,1,1])
    lambdas = np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 2, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 0, 0, 2, 0, 2, 1]])
    return lambdas.T, taus

def plot_lik(ll, file_name = None):
    x = np.arange(len(ll))
    plt.xlabel('Iterations')
    plt.ylabel('LL')
    plt.plot(x, ll)
    if file_name:
        plt.savefig(os.path.join('.',"%s.png" % file_name), bbox_inches="tight")
    else:
        plt.show()
    plt.close()

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

def plot_stability(lam_list, cyy_list, taus_list):
    lam_list = adjust_sign(lam_list)
    lam_02 = [lam[0][2] for lam in lam_list]
    lam_12 = [lam[1][2] for lam in lam_list]
    lam_22 = [lam[2][2] for lam in lam_list]
    x = np.arange(len(lam_list))

    #plot reference line
    true_lambdas, true_taus = load_stability_test_settings()
    true_lambdas = sort_factor_by_variance(np.transpose(varimax(true_lambdas)))
    plt.plot(x, [true_lambdas[0][2]]*len(x), color='b',ls='-')
    plt.plot(x, [true_lambdas[1][2]]*len(x), color='r',ls='-')
    plt.plot(x, [true_lambdas[2][2]]*len(x), color='k',ls='-')
    plt.plot(x, lam_02, color='b', ls='None', marker='.')
    plt.plot(x, lam_12, color='r', ls='None', marker='.')
    plt.plot(x, lam_22, color='k', ls='None', marker='.')
    # plt.plot(x, lam_22, color='g', ls='None', marker='.')
    plt.savefig(os.path.join('.', "stability_varimax_2.png"), bbox_inches="tight")
    plt.close()

def draw_iter_boxplot(ml_iters, em_iters, file_name=None):
    data = [ml_iters, em_iters]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data)
    ax.set_xticklabels(['ML', 'EM'])

    fig.savefig(os.path.join('.', "%s.png" % file_name), bbox_inches='tight')
    plt.close()

def adjust_sign(lam_list):
    new_lam_list = []
    for lam in lam_list:
        if(lam[0][0] < 0):
            lam[0] *= -1
        if(lam[1][1] < 0):
            lam[1] *= -1
        if(lam[2][2] < 0):
            lam[2] *= -1
        new_lam_list.append(lam)
    return new_lam_list


