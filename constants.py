import numpy as np
import scipy
sigma0 = np.matrix([[1,0],[0,1]])
sigma1 = np.matrix([[0,1],[1,0]])
sigma2 = np.matrix([[0,-1.j],[1.j,0]])
sigma3 = np.matrix([[1,0],[0,-1]])

sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
sqrt = np.sqrt
pi = np.pi
exp = np.exp
conj = np.conj

def factorial(x):
    return scipy.special.gamma(x+1)