# use S24 in stablization of Finite energy...

import numpy as np
import math
import mpmath
from scipy.special import hermite
from mpmath import factorial

# compute normalization factors and overlap factor
def Norm2(Delta,mu):
    t = np.tanh(Delta**2)
    s = np.sinh(Delta**2)
    c = np.cosh(Delta**2)
    x1 = mpmath.jtheta(3,2*np.pi*t*mu*1.j,np.exp(-4*np.pi*t))
    x2 = mpmath.jtheta(3,0,np.exp(-4*np.pi/t))
    x3 = mpmath.jtheta(3,2*np.pi*t*(1-mu)*1.j,np.exp(-4*np.pi*t))
    x4 = mpmath.jtheta(3,2.j/t,np.exp(-4*np.pi/t))
    return np.exp(-np.pi*t*mu)/np.sqrt(np.pi*(1-np.exp(-4*Delta**2)))*(x1*x2+np.exp(-np.pi/c/s)*x3*x4)

def overlap(Delta):
    t = np.tanh(Delta**2)
    s = np.sinh(Delta**2)
    c = np.cosh(Delta**2)
    x1 = float(mpmath.jtheta(3,t*np.pi*1.j,np.exp(-4*np.pi*t)).real)
    x2 = float(mpmath.jtheta(3,-np.pi*1.j/t,np.exp(-4*np.pi/t)).real)
    x3 = float(mpmath.jtheta(3,-t*np.pi*1.j,np.exp(-4*np.pi*t)).real)
    x4 = float(mpmath.jtheta(3,np.pi*1.j/t,np.exp(-4*np.pi/t)).real)
    return np.exp(-np.pi/4*(t+1/t))/np.sqrt(np.pi*(1-np.exp(-4*Delta**2)))*(x1*x2+x3*x4)

# overlap between \ket{n} and finite energy GKP state
def GKPstate_nBasis(Delta,mu,n):
    # result = exp(-Delta**2 * n) \sum_k \psi^*_n (2 pi**0.5 (k+mu/2)) where \psi_n is the n-th eigen state of harmonic ocillator
    def psi(n,x): # normalization factor
        N = 1./np.sqrt(np.sqrt(np.pi)*float(2**n)*float(factorial(n)))
        Hr=hermite(n)
        Psix = N*Hr(x)*np.exp(-0.5*x**2)
        return Psix
    cutoff = 5
    return np.exp(-Delta**2*n)*sum([psi(n,2*np.sqrt(np.pi)*(i + mu/2)) for i in range(-cutoff,cutoff)])


# compute N_\gamma(\ket{\mu}\ket{\nu}) , output a n_cut dimensional matrix
# n_cut is a cut off for phonon number
def N_gamma_normal_basis(mu, nu, Delta, gamma, n_cut, L_range):
    #L_range is lattice range
    #generate lattice points for mu and nu
    L_mu = [np.sqrt(np.pi/2)*((2*n1 + mu)+n2*1.j) 
            for n1 in range(-int(L_range/2),int(L_range/2)) 
            for n2 in range(-int(L_range),int(L_range))
        ]
    L_nu = [np.sqrt(np.pi/2)*((2*n1 + nu)+n2*1.j) 
            for n1 in range(-int(L_range/2),int(L_range/2)) 
            for n2 in range(-int(L_range),int(L_range))
        ]
    # compute matrix element
    result = np.zeros([n_cut,n_cut],dtype=complex)
    for i in range(n_cut):
        for j in range(n_cut):
            result[i,j] = sum([ 
                            np.exp(-(Delta**2+1/2)*(abs(a)**2+abs(ap)**2)) * np.exp(-1.j*(a.real*a.imag - ap.real*ap.imag)) * np.exp(gamma*a*ap.conjugate()) * a**i * (ap.conjugate())**j / np.sqrt(float(factorial(i)*factorial(j)))
                            for a in L_mu for ap in L_nu
                        ])
    return result

# use Pauli operator as input
def N_gamma_pauli_basis(pauli, Delta, gamma, n_cut, L_range):
    if pauli == 'I':
        return N_gamma_normal_basis(0,0, Delta, gamma, n_cut, L_range) + N_gamma_normal_basis(1,1, Delta, gamma, n_cut, L_range)
    elif pauli == 'X':
        return N_gamma_normal_basis(0,1, Delta, gamma, n_cut, L_range) + N_gamma_normal_basis(1,0, Delta, gamma, n_cut, L_range)
    elif pauli == 'Y':
        return -1.j*N_gamma_normal_basis(0,1, Delta, gamma, n_cut, L_range) + 1.j*N_gamma_normal_basis(1,0, Delta, gamma, n_cut, L_range)
    elif pauli == 'Z':
        return N_gamma_normal_basis(0,0, Delta, gamma, n_cut, L_range) - N_gamma_normal_basis(1,1, Delta, gamma, n_cut, L_range)
    else:
        raise TypeError('pauli operator type not found')

# general form, invoke this function recommended
def N_gamma(input,Delta,gamma,n_cut, L_range):
    if type(input) == str:
        return N_gamma_pauli_basis(input, Delta, gamma, n_cut, L_range)
    elif input in [(i,j) for i in range(2) for j in range(2)]:
        return N_gamma_normal_basis(input[0], input[1], Delta, gamma, n_cut, L_range)
    else:
        raise TypeError('input type error')

'''
# just for test
if __name__ == '__main__':
    n_ave = 2
    Delta = 1/np.sqrt(2*n_ave + 1)
    gamma = 0.1
    n_cut = 10
    L_range = 7
    res = N_gamma('I', Delta, gamma, n_cut, L_range)
    print(np.trace(res))
 '''   
if __name__ == '__main__':
    Delta = 0.481
    print(Norm2(Delta, 0))
    print(Norm2(Delta, 1.))
    print(overlap(Delta))
    mu0_nBasis = [GKPstate_nBasis(Delta,0,n) for n in range(70)]
    mu1_nBasis = [GKPstate_nBasis(Delta,1,n) for n in range(70)]
    print(np.linalg.norm(mu0_nBasis)**2)
    print(np.linalg.norm(mu1_nBasis)**2)
    print(np.dot(mu0_nBasis,mu1_nBasis))