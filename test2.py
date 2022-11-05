import numpy as np
import N_gamma
import mpmath
from constants import *
sqrt = np.sqrt

func1 = N_gamma.m

def eq_mu(Delta,mu,nu):
    #raise ValueError('not consistent in test2')
    if mu not in [0,1] or nu not in [0,1]:
        raise TypeError('mu ,nu must be 0 or 1')
    if mu ==nu:
        def theta(z,t):
            return mpmath.jtheta(3,pi*z,exp(1.j*pi*t))
        t = np.tanh(Delta**2)
        s = np.sinh(Delta**2)
        c = np.cosh(Delta**2)
        x1 = theta(0,4.j/t)
        x2 = theta(2.j*mu*t,4.j*t)
        x3 = theta(2.j/t,4.j/t)
        x4 = theta(2.j*(1-mu)*t,4.j*t)
        res = np.exp(-np.pi*t*mu)/np.sqrt(np.pi*(1-np.exp(-4*Delta**2)))*(x1*x2+np.exp(-np.pi/c/s)*x3*x4)
    else:
        t = np.tanh(Delta**2)
        s = np.sinh(Delta**2)
        c = np.cosh(Delta**2)
        x1 = mpmath.jtheta(3,t*np.pi*1.j,np.exp(-4*np.pi*t))
        x2 = mpmath.jtheta(3,-np.pi*1.j/t,np.exp(-4*np.pi/t))
        x3 = mpmath.jtheta(3,-t*np.pi*1.j,np.exp(-4*np.pi*t))
        x4 = mpmath.jtheta(3,np.pi*1.j/t,np.exp(-4*np.pi/t))
        res = np.exp(-np.pi/4*(t+1/t))/np.sqrt(np.pi*(1-np.exp(-4*Delta**2)))*(x1*x2+x3*x4)
    return float(res.real)
    

if __name__ == '__main__':
    func1 = N_gamma.m
    func2 = eq_mu
    Delta = 1
    mu = 0
    nu = 0
    print('func1 =',func1(Delta,mu,nu)-func2(Delta,mu,nu))
    print('=',func1(Delta,mu,nu))
    print('=',func2(Delta,mu,nu))
    mu = 1
    nu = 1
    print('func1 =',func1(Delta,mu,nu)-func2(Delta,mu,nu))
    mu = 1
    nu = 0
    print('func1 =',func1(Delta,mu,nu)-func2(Delta,mu,nu))
    mu = 0
    nu = 1
    print('func1 =',func1(Delta,mu,nu)-func2(Delta,mu,nu))