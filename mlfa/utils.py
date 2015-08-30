import numpy as np
from scipy import linalg

def woodbury(lambdas, taus):
    """
        woodbury identity util function for inv(taus + transpose(lambdas)*lambdas)
    """
    q = lambdas.shape[0]
    taus_inv = linalg.inv(taus)
    left = taus_inv.dot(np.transpose(lambdas))
    return taus_inv - left.dot( linalg.inv(np.identity(q) + np.dot(lambdas,taus_inv).dot(np.transpose(lambdas))) ).dot(np.transpose(left))

def get_index_from_factor_pattern(pattern):
    """
        get the matrix index from the factor pattern which is a matrix with 1s and 0s, q*p
    """
    return [[i for i, v in enumerate(row) if v ] for row in np.transpose(pattern)]

