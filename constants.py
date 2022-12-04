import numpy as np
import scipy
from scipy.special import eval_genlaguerre , loggamma
from functools import lru_cache
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
ln = np.log

def factorial(x):
    return scipy.special.gamma(float(x)+1)

# compute n_Delta
def comp_n_Delta(Delta):
    return 1/(exp(2*Delta**2)-1)

'''
# eval_genlaguerre with recurrance relation
# https://en.wikipedia.org/wiki/Laguerre_polynomials
@lru_cache(1000)
def my_eval_genlaguerre(n,alpha,x):
    assert type(n) == int
    assert n >= 0
    if n ==0:
        return 1
    elif n == 1:
        return -x+alpha+1
    else:
        return (2+(alpha-1-x)/n)*my_eval_genlaguerre(n-1,alpha,x) - (1+(alpha-1)/n)*my_eval_genlaguerre(n-2,alpha,x)


@lru_cache(1000)
def my_eval_genlaguerre(n,a,x):
    return eval_genlaguerre(n,a,x)
'''

def lDl(l,lp,alpha):
    assert type(l) == int
    assert type(lp) == int
    if alpha == 0:
        if l != lp:
            return 0
        else:
            fac = 0
    else:
        fac = (l-lp)*ln(alpha)
    if l>=lp:
        res = exp(-0.5*abs(alpha)**2 + 0.5*(loggamma(lp+1)-loggamma(l+1))+fac)*eval_genlaguerre(lp,l-lp,abs(alpha)**2)
    else:
        l,lp=lp,l
        res = lDl(l,lp,alpha)
        res = res.conjugate()
        l,lp=lp,l
    return res

if __name__ == '__main__':
    gamma = 0.1
    alpha = sqrt(pi/2/gamma)
    print(lDl(1400,1100,alpha))
    '''
    for l in range(1000):
        for lp in range(l,1000):
            print(l,lp,lDl(l, lp, alpha))
    '''