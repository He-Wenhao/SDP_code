import numpy as np
from N_gamma import * # characterize amplitude damping channel N_gamma
import cvxpy as cp # https://www.cvxpy.org/examples/basic/sdp.html
from constants import *
import warnings
import scipy
#warnings.filterwarnings("error")

scale = 1

# data structure for GKP code under amplitude damping problem
class GKP_ampDamp:
    def __init__(self,gamma,Delta):
        self.n_ave = 0.5*(1/(Delta**2) - 1) # average phonon number
        self.gamma = gamma # damping rate gamma for amplitude damping channel
        self.Delta = Delta
        self.Recovery = None # Choi matrix for recovery channel

    # find an optimized recovery with SDP
    # use phonon number basis
    def optimize_Recovery_numberBasis(self,n_cut):
        # the SDP problem is
        # max_R Tr 1/4 + 1/4 \sum_{i=x,y,z} A_i N_\gamma (\sigma_i)
        # R = 1/2 I_n \otimes I_code + \sum_{i=x,y,z} A_i \otimes \sigma_i >= 0
        #compute N_gamma_othNor_pauli(sigma_i)
        N_gamma_0 = N_gamma_othNor_pauli('I', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_1 = N_gamma_othNor_pauli('X', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_2 = N_gamma_othNor_pauli('Y', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_3 = N_gamma_othNor_pauli('Z', self.Delta, self.gamma, n_cut).transpose()
        K = np.kron(N_gamma_0, sigma0) + np.kron(N_gamma_1, sigma1) + np.kron(N_gamma_2, sigma2) + np.kron(N_gamma_3, sigma3)
        K = K.real# K should be real
        # A_i s are Hermitian
        R = cp.Variable((n_cut*2,n_cut*2), symmetric=True)

        # compute partial trace of R
        constraints = [R >> 0]
        for i1 in range(n_cut):
            for j1 in range(n_cut):
                cons = np.zeros([n_cut*2,n_cut*2])
                for k in range(2):
                    cons[i1*2+k,j1*2+k] = 1
                constraints += [cp.trace(cons@R)==np.eye(n_cut)[i1,j1]]

                        
        prob = cp.Problem(
            cp.Maximize(1/8*cp.trace(K@R)*scale),
            constraints
        )

        
        # optimization---')
        prob.solve(eps=1e-7)
        #print('---end optimization---')
        # result choi matrix 
        #choi = 1/2*cp.kron(A0,np.identity(2)) + cp.kron(A1, sigma1)+ cp.kron(A2, sigma2)+ cp.kron(A3, sigma3) 
        choi = None
        return prob.value/scale, choi

    def transpose_0th(self,i_cut):
        gamma = self.gamma
        n_ave = self.n_ave
        factor = np.sqrt((gamma*n_ave+1)/(gamma*n_ave))
        res = 0
        # the SDP problem is
        # max_R Tr 1/4 + 1/4 \sum_{i=x,y,z} A_i N_\gamma (\sigma_i)
        # R = 1/2 I_n \otimes I_code + \sum_{i=x,y,z} A_i \otimes \sigma_i >= 0
        #compute N_gamma_othNor_pauli(sigma_i)
        N_gamma_0 = N_gamma_othNor_pauli('I', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_1 = N_gamma_othNor_pauli('X', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_2 = N_gamma_othNor_pauli('Y', self.Delta, self.gamma, n_cut).transpose()
        N_gamma_3 = N_gamma_othNor_pauli('Z', self.Delta, self.gamma, n_cut).transpose()
        N = np.kron(N_gamma_0, sigma0) + np.kron(N_gamma_1, sigma1) + np.kron(N_gamma_2, sigma2) + np.kron(N_gamma_3, sigma3)
        N = N.real
        K_gamma_0 = N_gamma_othNor_pauli('I', self.Delta, self.gamma, n_cut,factor).transpose()
        K_gamma_1 = N_gamma_othNor_pauli('X', self.Delta, self.gamma, n_cut,factor).transpose()
        K_gamma_2 = N_gamma_othNor_pauli('Y', self.Delta, self.gamma, n_cut,factor).transpose()
        K_gamma_3 = N_gamma_othNor_pauli('Z', self.Delta, self.gamma, n_cut,factor).transpose()
        K = np.kron(K_gamma_0, sigma0) + np.kron(K_gamma_1, sigma1) + np.kron(K_gamma_2, sigma2) + np.kron(K_gamma_3, sigma3)
        K = K.real# K should be real
        # normalize partial trace of K
        p_K = np.zeros([n_cut,n_cut])
        for i1 in range(n_cut):
            for j1 in range(n_cut):
                for k in range(2):
                    p_K[i1,j1] += K[i1*2+k,j1*2+k]
        #print(p_K)
        evalues, evectors = np.linalg.eig(a)
        # keep only large square root matrix exists
        sqrt_matrix = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
        K_neg_hal = scipy.linalg.sqrtm(np.linalg.inv(p_K))
        K_neg_hal = np.kron(K_neg_hal, sigma0)
        K = K_neg_hal@K@K_neg_hal
        #print(1/8 * np.trace(np.matmul(K,N)))

if __name__ == '__main__':
    gamma = 0.
    Delta = 0.481
    n_cut = 40
    i_cut = 10
    gkp = GKP_ampDamp(gamma,Delta)
    res = 1-gkp.optimize_Recovery_numberBasis(n_cut)[0]
    print(res)
    
    #gkp.transpose_0th(i_cut)
'''
0.309 (1)
-9.256075101937711e-07
4.181980572948163e-06
3.896591615193401e-05
3.553276150192186e-05
7.585589813952076e-05
0.00011631823934021845
0.00020295767959865874
0.0003212432188298697
0.0005225139607303309
0.00079113320735269
0.0011354625287904874

0.309 (2)
-7.461103281869441e-07
-3.228504963370682e-07
4.082334491739559e-05
2.9547743647606595e-05
7.194565866841529e-05
0.000110661739531559
0.0001869747794781551
0.00029093608016050876
0.0005083193731896252
0.0007606022792172595
0.0011078870684045894
    '''

