from __future__ import division
import sys
from collections import defaultdict
from mfa import MixtureFA
from cluster import MFACluster
from ..modules.simulate import FactorModel
import os
import numpy as np
import csv
import cPickle

"""
    what if we fit a 2 mixtures?
"""
def load_params():
    mu = np.array([0,0,0])
    lambdas = np.array([[1,1,1]])
    taus = np.sqrt([0.1,0.1,0.1])
    return mu, lambdas.T, taus

def get_sample(n_obs, p, seed = 5):
    """
        create 1000 obs with 3 mixtures of ratio 1:3:6
    """
    sample_sizes = get_indices(n_obs, p, seed)
    print(sample_sizes)
    n1 = sample_sizes[0]
    n2 = sample_sizes[1]
    n3 = sample_sizes[2]
    mu, lambdas, std = load_params()
    fm1 = FactorModel(n1, lambdas, std, mu=mu, seed=seed)
    fm2 = FactorModel(n2, lambdas, std, mu=mu + 10, seed=seed)
    fm3 = FactorModel(n3, lambdas, std, mu=mu + 20, seed=seed)
    fm1.simulate()
    fm2.simulate()
    fm3.simulate()
    Y = [1] * n1 + [2] * n2 + [3] * n3
    return np.vstack((fm1.obs, fm2.obs, fm3.obs)), np.array(Y)

def get_indices(n_obs, p, seed):
    """generate sample sizes for the 3 mixtures according to the predefined proportions"""
    np.random.seed(seed)
    com_id, sample_sizes = np.unique(np.random.choice(3, n_obs, p=p),return_counts=True)
    return sample_sizes

def cal_ratio(label):
    dic = defaultdict(int)
    for l in label:
        dic[l] += 1
    return [n / len(label) for n in dic.values()]

def sample_generator(iters, p, size, seed=None):
    """
        iters : number of simulations
        size : sample size in each simulation
    """
    for i in xrange(iters):
        s = seed if seed is not None else i
        X, Y = get_sample(size, p, s)
        yield X, Y

def mfa_experiment(n_mixtures, output=False):
    p = [0.1, 0.3, 0.6]
    mus = []
    pis = []
    pis_diff = []
    phis = []
    lls = []
    lambdas = []
    count = 0
    for X, ratio in sample_generator(1, p, 100):
        count += 1
        mfa = MixtureFA(n_mixtures, 1)
        mfa.fit(X)

        # np.savetxt("data.csv", X, delimiter=',')

        mus.append(mfa.mu)
        pis.append(mfa.pi)
        phis.append(mfa.phi)
        pis_diff.append(np.abs(mfa.pi - np.array(ratio)))
        lambdas.append(np.array([l.dot(l.T) for l in np.vsplit(mfa.lambdas,n_mixtures)]))
        # pdb.set_trace()
        lls.append(mfa.ll[-1])
        if count % 10 == 0:
            print("finished %d" % count)
    print mfa.mu
    print mfa.pi
    if output:
        with open(os.path.join('.','result_seed0.txt'), 'w') as outfile:
            outfile.write("Mean of MU\n")
            outfile.write(np.array2string(np.mean(np.array(mus), axis=0), separator=',') + '\n')
            # outfile.write("std of MU\n")
            # outfile.write(np.array2string(np.std(np.array(mus), axis=0), separator=',') + '\n')
            outfile.write("Mean of lambda.T.dot(lambda)\n")
            outfile.write(np.array2string(np.mean(np.array(lambdas), axis=0), separator=',') + '\n')
            # outfile.write("std of lambda.T.dot(lambda)\n")
            # outfile.write(np.array2string(np.std(np.array(lambdas), axis=0), separator=',') + '\n')
            outfile.write("Mean of Phi\n")
            outfile.write(np.array2string(np.mean(np.array(phis), axis=0), separator=',') + '\n')
            # outfile.write("std of Phi\n")
            # outfile.write(np.array2string(np.std(np.array(phis), axis=0), separator=',') + '\n')
            outfile.write("Mean of Pi\n")
            outfile.write(np.array2string(np.mean(np.array(pis), axis=0), separator=',') + '\n')
            # outfile.write("std of Pi\n")
            # outfile.write(np.array2string(np.std(np.array(pis), axis=0), separator=',') + '\n')

def cluster_experiment():
    p = [0.33,0.33,0.34]
    max_k = 5
    n_factors = 1
    res = {'cv':np.zeros(max_k-1),'ca':np.zeros(max_k-1),'cstd':np.zeros(max_k-1),'aic':np.zeros(max_k-1),'bic':np.zeros(max_k-1)}
    result_list = []
    count = 100
    distance_dist = np.zeros((1, max_k-1),dtype=int)
    for X, Y in sample_generator(2, p, 600):
        print count
        result = {'id':count}
        # recorde the input data
        # np.savetxt("intput_data_%d.csv" % count, X, delimiter=',')
        # fit mfa model with validation
        mfa_cluster = MFACluster(X,max_k,n_factors,Y,iters=10)
        mfa_cluster.fit()
        res['cv'][mfa_cluster.best_k("voting")-2] += 1
        res['ca'][mfa_cluster.best_k("averaging")-2] += 1
        res['cstd'][mfa_cluster.best_k("std")-2] += 1

        result['cv'] = mfa_cluster.best_k("voting")
        result['ca'] = mfa_cluster.best_k("averaging")
        result['cstd'] = mfa_cluster.best_k("std")
        distance_dist = np.vstack((distance_dist, mfa_cluster.result_matrix))

        # fit mfa model to whole dataset with aic and bic

        min_aic = min_bic = sys.maxint
        k_aic = k_bic = 0
        for k in xrange(2, max_k+1):
            m = MixtureFA(k, n_factors)
            m.fit(X)
            result['lik_%d' % k] = m.ll[-1]
            result['aic_%d' % k] = m.aic()
            result['bic_%d' % k] = m.bic()
            if m.aic() < min_aic:
                k_aic = k
                min_aic = m.aic()
            if m.bic() < min_bic:
                k_bic = k
                min_bic = m.bic()
        res['aic'][k_aic-2] += 1
        res['bic'][k_bic-2] += 1
        result['aic'] = k_aic
        result['bic'] = k_bic

        # collect result
        # save iter number, likelihood_k, aic_k, bic_k, cv, ca, cstd into csv file
        result_list.append(result)
        count += 1
    save_dict_to_csv(result_list)
    np.savetxt(os.path.join('.','result',"distance_dist.csv"), distance_dist, delimiter=',')
    print res

def cluster_simulation(ids, seed):
    """This will run single simulation with seed and id arguments.
    The result will be serialized into 2 pickle files, one distance matrix and the other one for detailed result
    """
    p = [0.33,0.33,0.34]
    max_k = 5
    n_factors = 1
    result = {'id':ids}
    for X, Y in sample_generator(1, p, 100, seed = seed):
        mfa_cluster = MFACluster(X,max_k,n_factors,Y,iters=10)
        mfa_cluster.fit()
        result['cv'] = mfa_cluster.best_k("voting")
        result['ca'] = mfa_cluster.best_k("averaging")
        result['cstd'] = mfa_cluster.best_k("std")

        min_aic = min_bic = sys.maxint
        k_aic = k_bic = 0
        for k in xrange(2, max_k+1):
            m = MixtureFA(k, n_factors)
            m.fit(X)
            result['lik_%d' % k] = m.ll[-1]
            result['aic_%d' % k] = m.aic()
            result['bic_%d' % k] = m.bic()
            if m.aic() < min_aic:
                k_aic = k
                min_aic = m.aic()
            if m.bic() < min_bic:
                k_bic = k
                min_bic = m.bic()
        result['aic'] = k_aic
        result['bic'] = k_bic
    cPickle.dump(result, open("pickle/%s_result.p" % ids, 'w'))
    cPickle.dump(mfa_cluster.result_matrix, open("pickle/%s_matrix.p" % ids, 'w'))

def save_dict_to_csv(result_list):
    with open(os.path.join('.','result',"result2.csv"),'wb') as f:
        writer = csv.DictWriter(f, result_list[0].keys())
        writer.writeheader()
        for result in result_list:
            writer.writerow(result)

if __name__ == "__main__":
    cluster_simulation(1,6)
