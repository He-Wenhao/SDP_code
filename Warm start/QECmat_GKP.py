#import warnings
#warnings.filterwarnings("error")
import numpy as np
from scipy.special import hermite,  eval_genlaguerre, loggamma
import scipy

sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
sqrt = np.sqrt
pi = np.pi
exp = np.exp
conj = np.conj
ln = np.log


# generate m and M matrix for GKP()
class QECmat_GKP:
    def __init__(self,Delta,gamma,l_cut,m_sum_cutoff=20,M_sum_cutoff=5):
        self.gamma = gamma # damping rate gamma for amplitude damping channel
        self.Delta = Delta  # characterize energy
        self.n_Delta = 1/(exp(2*Delta**2)-1) # approximate average phonon number
        self.l_cut = l_cut  # how much kraus operator taking into consideration # typically 20
        self.m_sum_cutoff = m_sum_cutoff    # typically 20
        self.M_sum_cutoff = M_sum_cutoff    # typically 5
    
    # overlap of GKP code basis
    def m(self,keep_real = True):
        cutoff = self.m_sum_cutoff
        Delta = self.Delta
        cutoff = 20
        res = np.zeros((2,2),dtype=complex)
        for mu1 in [0,1]:
            for mu2 in [0,1]:
                for n1 in range(-cutoff,cutoff):
                    for n2 in range(-cutoff,cutoff):
                        Lambda = np.sqrt(np.pi/2)*(2*n1+mu1-mu2+1.j*n2)
                        res[mu1,mu2] += 1/(2*sqrt(np.pi)*(1-np.exp(-2*Delta**2)))* np.exp(1.j*np.pi*(n1+(mu1+mu2)/2)*n2)*np.exp(-1/(2*np.tanh(Delta**2))*abs(Lambda)**2) 
        # keep only real part
        if keep_real == True:
            res = res.real
        return np.matrix(res)

    # average photon number(rigorous result)
    def n_ave(self,sum_cutoff):
        Delta = self.Delta
        n_D = self.n_Delta
        factor = 0.5 / (2*sqrt(pi)*(1-exp(-2*Delta**2)))
        m0 = np.matrix([[m(Delta,mu,nu) for nu in [0,1]] for mu in [0,1]])
        cutoff = 20
        def K(Delta,mu,mup,cutoff):
            res = 0
            t2 = np.tanh(Delta**2)
            s2 = np.sinh(Delta**2)
            for n1 in range(-sum_cutoff,sum_cutoff):
                for n2 in range(-sum_cutoff,sum_cutoff):
                    Lambda = sqrt(pi/2)*(2*n1+mu-mup+n2*1.j)
                    alpha = exp(Delta**2)/sqrt(gamma+1/n_D)*conj(Lambda)
                    res += exp(-pi/(2*t2)*abs(Lambda)**2)*exp(1.j*pi*(n1+(mu+mup)/2)*n2)* abs(Lambda)**2 / 4 / s2**2
            return res
        K0 = np.matrix([[K(Delta,mu,nu,cutoff) for nu in [0,1]] for mu in [0,1]])
        return n_D - factor*np.trace(np.linalg.inv(m0)@K0)

    def _lDl(self,l,lp,alpha):
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
            res = self._lDl(l,lp,alpha)
            res = res.conjugate()
            l,lp=lp,l
        return res

    # error correction matrix
    # metric of basis E_l \ket{mu}, i.e. describe overlap and norm of GKP E_l|0> E_l|1> , for l = 0,1,2,...
    # return a 2l_cutoff * 2l_cutoff dimentional matrix
    def _M(self,keep_real = True):
        sum_cutoff = self.M_sum_cutoff
        l_cutoff = self.l_cut
        Delta = self.Delta
        gamma = self.gamma
        res = np.zeros((2*l_cutoff,2*l_cutoff),dtype=complex)
        # element of M matrix
        def M_ele(cutoff,l,mu,lp,mup):
            n_D = 1/(exp(2*Delta**2)-1)
            factor = (gamma*n_D)**((l+lp)/2)/(gamma*n_D+1)**((l+lp)/2+1) / (2*sqrt(pi)*(1-exp(-2*Delta**2)))
            res = 0
            t = np.tanh(Delta**2/2)
            for n1 in range(-cutoff,cutoff):
                for n2 in range(-cutoff,cutoff):
                    Lambda = sqrt(pi/2)*(2*n1+mu-mup+n2*1.j)
                    alpha = exp(Delta**2)/sqrt(gamma+1/n_D)*conj(Lambda)
                    lDl = self._lDl(l,lp,alpha)
                    res += exp(-(1-gamma)/2/(gamma+1/n_D)*abs(Lambda)**2)*exp(1.j*pi*(n1+(mu+mup)/2)*n2)*lDl
            return (factor*res).real
        for i in range(2*l_cutoff):
            for j in range(2*l_cutoff):
                res[i,j] = M_ele(sum_cutoff, i//2, i%2, j//2, j%2)
        # keep only real part
        if keep_real == True:
            res = res.real
        return res
    
    # transform into orthogonalized and normalized GKP basis
    def orth_M(self):
        Delta = self.Delta
        gamma = self.gamma
        Mmat = self._M()
        mmat = self.m()
        u, s, vh = np.linalg.svd(mmat, full_matrices=True)
        U = np.diag(np.array(s)**(-0.5))@vh
        U = np.kron(np.eye(self.l_cut), U)
        Mmat = U@Mmat@U.transpose()
        #print(U@Mmat@U.transpose())
        return Mmat
