# use S24 in stablization of Finite energy...
import warnings
warnings.filterwarnings("error")
import numpy as np
import math
import mpmath
from scipy.special import hermite,  eval_genlaguerre
import scipy
#from mpmath import factorial
from functools import lru_cache
from constants import *
def factorial(x):
    return scipy.special.gamma(x+1)

# metric of basis \ket{mu}
def m(Delta,mu1,mu2):

    cutoff = 20
    
    
    res = 0
    for n1 in range(-cutoff,cutoff):
        for n2 in range(-cutoff,cutoff):
            Lambda = np.sqrt(np.pi/2)*(2*n1+mu1-mu2+1.j*n2)
            res += 1/(2*sqrt(np.pi)*(1-np.exp(-2*Delta**2)))* np.exp(1.j*np.pi*(n1+(mu1+mu2)/2)*n2)*np.exp(-1/(2*np.tanh(Delta**2))*abs(Lambda)**2) 
    return res

# overlap between \ket{n} and finite energy GKP state
@lru_cache(1000)
def GKPstate_nBasis(Delta,mu,n):
    # result = exp(-Delta**2 * n) \sum_k \psi^*_n (2 pi**0.5 (k+mu/2)) where \psi_n is the n-th eigen state of harmonic ocillator
    def psi(n,x): # normalization factor
        N = 1./np.sqrt(np.sqrt(np.pi))*1/np.sqrt(np.exp2(n))*1/np.sqrt(factorial(n))
        Hr=hermite(n)
        Psix = N*Hr(x)*np.exp(-0.5*x**2)
        return Psix
    cutoff = 5
    return np.exp(-Delta**2*n)*sum([psi(n,2*np.sqrt(np.pi)*(i + mu/2)) for i in range(-cutoff,cutoff)])


def get_othNor_basis(Delta,n_cut):
    check = 0
    # generate GKP 0 and 1
    GKP0 = np.array([GKPstate_nBasis(Delta,0,n) for n in range(n_cut)])
    GKP1 = np.array([GKPstate_nBasis(Delta,1,n) for n in range(n_cut)])
    if check:
        print('standard:')
        print('norm0',m(Delta, 0,0))
        print('norm1',m(Delta, 1,1))
        print('overlap',m(Delta,0,1))
        print('numerical:')
        print('norm0',np.linalg.norm(GKP0)**2)
        print('norm1',np.linalg.norm(GKP1)**2)
        print('overlap',np.dot(GKP0,GKP1))
    # obtain an normalized and orthogonal basis
    norm0 = np.linalg.norm(GKP0)**2
    norm1 = np.linalg.norm(GKP1)**2
    overlap = np.dot(GKP0,GKP1)
    basis0 = GKP0/np.sqrt(norm0)
    basis1 = GKP1 - overlap/norm0*GKP0
    basis1 = basis1/np.linalg.norm(basis1)
    return [basis0,basis1]

# conpute element \bra{n} E_l \ket{\mu} with finite energy(\Delta)
@lru_cache(1000)
def vec_l_mu(n,l,mu,Delta,gamma):
    return gamma**(l/2) * (1-gamma)**(n/2) * np.sqrt(float(math.comb(n+l, n))) * GKPstate_nBasis(Delta,mu,n+l)

# inner product matrix
def M_inner(l1,mu1,l2,mu2,Delta,n_cut,gamma):
    return sum([vec_l_mu(n,l1,mu1,Delta,gamma)*vec_l_mu(n,l2,mu2,Delta,gamma) for n in range(n_cut)])

def test_completeness():
    n_cut = 50
    Delta = 0.309
    gamma = 0.1
    M = np.zeros([n_cut*2,n_cut*2])
    for l1 in range(n_cut):
        for l2 in range(n_cut):
            for mu1 in range(2):
                for mu2 in range(2):
                    M[l1*2+mu1,l2*2+mu2] = M_inner(l1,mu1,l2,mu2,Delta,n_cut,gamma)
    a,b = np.linalg.eig(M)
    print('M matrix:',a)
    # construct 'identity' matrix for the support of E_i \ket{mu} 
    v = np.zeros([n_cut,n_cut*2])
    for n in range(n_cut):
        for l in range(n_cut):
            for mu in range(2):
                v[n,l*2+mu] = vec_l_mu(n,l,mu,Delta,gamma)
    I = v@np.linalg.inv(M)@v.transpose()
    a , b = np.linalg.eig(I)
    print('identity', a)





# compute N_\gamma(\ket{\mu}\ket{\nu}) , output a n_cut dimensional matrix
# n_cut is a cut off for phonon number
# with orthogonal and normalized basis
def N_gamma_othNor(mu, nu, Delta, gamma, n_cut,factor = 1):
    basis = get_othNor_basis(Delta,n_cut)
    result = np.zeros([n_cut,n_cut])
    for i in range(n_cut):
        for j in range(n_cut):
            l_range = n_cut - max(i,j)
            result[i,j] = sum([
                (gamma**l)*np.sqrt(float(math.comb(i+l, l)*math.comb(j+l, l)))*((1-gamma)**((i+j)/2)) * basis[mu][i+l]*basis[nu][j+l] * (factor **l)
                for l in range(l_range)
            ])
    return result

        
# use Pauli operator as input
def N_gamma_othNor_pauli(pauli, Delta, gamma, n_cut, factor = 1):
    if pauli == 'I':
        return N_gamma_othNor(0,0, Delta, gamma, n_cut, factor) + N_gamma_othNor(1,1, Delta, gamma, n_cut, factor)
    elif pauli == 'X':
        return N_gamma_othNor(0,1, Delta, gamma, n_cut, factor) + N_gamma_othNor(1,0, Delta, gamma, n_cut, factor)
    elif pauli == 'Y':
        return -1.j*N_gamma_othNor(0,1, Delta, gamma, n_cut, factor) + 1.j*N_gamma_othNor(1,0, Delta, gamma, n_cut, factor)
    elif pauli == 'Z':
        return N_gamma_othNor(0,0, Delta, gamma, n_cut, factor) - N_gamma_othNor(1,1, Delta, gamma, n_cut, factor)
    else:
        raise TypeError('pauli operator type not found')

def test_GKPstate_nBasis(Delta):
    print(m(Delta, 0,0))
    print(m(Delta, 1,1))
    print(m(Delta,0,1))
    mu0_nBasis = [GKPstate_nBasis(Delta,0,n) for n in range(70)]
    mu1_nBasis = [GKPstate_nBasis(Delta,1,n) for n in range(70)]
    print(np.linalg.norm(mu0_nBasis)**2)
    print(np.linalg.norm(mu1_nBasis)**2)
    print(np.dot(mu0_nBasis,mu1_nBasis))

def M(Delta,gamma,cutoff,l,mu,lp,mup):
    n_D = 1/(exp(2*Delta**2)-1)
    factor = (gamma*n_D)**((l+lp)/2)/(gamma*n_D+1)**((l+lp)/2+1) / (2*sqrt(pi)*(1-exp(-2*Delta**2)))
    res = 0
    t = np.tanh(Delta**2/2)
    for n1 in range(-cutoff,cutoff):
        for n2 in range(-cutoff,cutoff):
            Lambda = sqrt(pi/2)*(2*n1+mu-mup+n2*1.j)
            alpha = exp(Delta**2)/sqrt(gamma+1/n_D)*conj(Lambda)
            if l>=lp:
                lDl = exp(-abs(alpha)**2/2)*sqrt(factorial(lp)/factorial(l))*eval_genlaguerre(lp,l-lp,abs(alpha)**2)*alpha**(l-lp)
            else:
                l,lp=lp,l
                lDl = exp(-abs(alpha)**2/2)*sqrt(factorial(lp)/factorial(l))*eval_genlaguerre(lp,l-lp,abs(alpha)**2)*alpha**(l-lp)
                lDl = lDl.conjugate()
                l,lp=lp,l
            res += exp(-(1-gamma)/2/(gamma+1/n_D)*abs(Lambda)**2)*exp(1.j*pi*(n1+(mu+mup)/2)*n2)*lDl

    return (factor*res).real

def n_Delta(Delta):
    return 1/(exp(2*Delta**2)-1)

def n_ave(Delta):
    n_D = 1/(exp(2*Delta**2)-1)
    factor = 0.5 / (2*sqrt(pi)*(1-exp(-2*Delta**2)))
    m0 = np.matrix([[m(Delta,mu,nu) for nu in [0,1]] for mu in [0,1]])
    cutoff = 20
    def K(Delta,mu,mup,cutoff):
        res = 0
        t2 = np.tanh(Delta**2)
        s2 = np.sinh(Delta**2)
        for n1 in range(-cutoff,cutoff):
            for n2 in range(-cutoff,cutoff):
                Lambda = sqrt(pi/2)*(2*n1+mu-mup+n2*1.j)
                alpha = exp(Delta**2)/sqrt(gamma+1/n_D)*conj(Lambda)
                res += exp(-pi/(2*t2)*abs(Lambda)**2)*exp(1.j*pi*(n1+(mu+mup)/2)*n2)* abs(Lambda)**2 / 4 / s2**2
        return res
    K0 = np.matrix([[K(Delta,mu,nu,cutoff) for nu in [0,1]] for mu in [0,1]])
    return n_D - factor*np.trace(np.linalg.inv(m0)@K0)

def orth_M(Delta,gamma,dimL,cutoff):
    Mmat = np.matrix([[M(Delta,gamma,cutoff,i//2,i%2,j//2,j%2) for j in range(dimL*2)] for i in range(dimL*2)])
    mmat = np.matrix([[m(Delta,mu1,mu2) for mu2 in [0,1]] for mu1 in [0,1]])
    u, s, vh = np.linalg.svd(mmat, full_matrices=True)
    U = np.diag(np.array(s)**(-0.5))@vh
    U = np.kron(np.eye(dimL), U)
    Mmat = U@Mmat@U.transpose()
    #print(U@Mmat@U.transpose())
    return Mmat


def tranpose_fid(Delta,gamma,dimL,cutoff):
    #print('n_Delta',n_Delta(Delta))
    #print('n_ave',n_ave(Delta))
    #test_GKPstate_nBasis(Delta)
    #print('orth_M test',orth_M(Delta, 0, 3,5))
    Msqrt = scipy.linalg.sqrtm(orth_M(Delta,gamma,dimL,cutoff))
    ptrMsqrt = Msqrt.reshape([dimL,2,dimL,2])
    ptrMsqrt = np.matrix([[ptrMsqrt[i,0,j,0]+ptrMsqrt[i,1,j,1] for j in range(dimL)] for i in range(dimL)])
    fid = 0.25*np.trace(ptrMsqrt@ptrMsqrt.transpose())
    #print(1-fid)
    return 1-fid
    '''
    for l in range(4):
        for mu in [0,1]:
            lp=l
            mup=mu
            print('l mu lp mup =',l,mu,lp,mup)
            print('func =',M1(Delta,gamma,l,mu,lp,mup))
            lmu = [vec_l_mu(n,l,mu,Delta,gamma) for n in range(80)]
            lpmup = [vec_l_mu(n,lp,mup,Delta,gamma) for n in range(80)]
            print('brute =',np.dot(lmu,lpmup))
    '''
    
if __name__ == '__main__':
    data=[]
    for gamma in np.linspace(0,0.1,11):
        Delta = 0.481
        dimL=20
        cutoff = 5
        res = tranpose_fid(Delta, gamma, dimL, cutoff)
        data.append(res.real)
        print(res)
    print(data)

    '''
    0.309
    -4.3655026366604515e-31j
(1.469043408364179e-06-2.4492108227927982e-31j)
(6.827167515544019e-06-2.52612666430367e-31j)
(2.0652076215710302e-05-3.394304345525873e-31j)
(4.9890094309668065e-05-2.6286341040351844e-31j)
(0.00010406770167115109-2.672703781796637e-31j)
(0.00019533716155106795-2.6999529516651377e-31j)
(0.0003383881575876879-3.3839667591837584e-31j)
(0.0005502794177317805-2.7797636620002345e-31j)
(0.0008502534286682906-2.8186528364252067e-31j)
(0.0012595910770020202-2.848453771883159e-31j)

0.481

(1.3322676295501878e-15-1.8087116311980755e-19j)
(0.0007032768163517389-1.9430656486476053e-18j)
(0.001578879275889955-2.762900891937408e-19j)
(0.002638626255128451-2.6470842000220957e-19j)
(0.003894026091168401+1.429265967351255e-18j)
(0.005356332498326677-1.94806208400381e-18j)
(0.007036571652055845-3.0248969270292773e-19j)
(0.008945546653714165-5.535255256174445e-19j)
(0.011093824705171396-1.8190979266271063e-19j)
(0.013491711403485374+1.2636226158023207e-18j)
(0.016149215699938435+1.143643512440785e-16j)
    '''