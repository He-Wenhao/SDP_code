import numpy as np
from N_gamma import * # characterize amplitude damping channel N_gamma
import cvxpy as cp # https://www.cvxpy.org/examples/basic/sdp.html
from constants import *


# data structure for GKP code under amplitude damping problem
class GKP_ampDamp:
    def __init__(self,gamma,Delta):
        self.n_ave = None # average phonon number
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
        S0 = np.zeros([n_cut,n_cut])
        S0[0,0] = 1
        S0[1,1] = 1
        S1 = np.zeros([n_cut,n_cut])
        S1[0,1] = 1
        S1[1,0] = 1
        S2 = np.zeros([n_cut,n_cut],dtype=complex)
        S2[0,1] = -1.j
        S2[1,0] = 1.j
        S3 = np.zeros([n_cut,n_cut])
        S3[0,0] = 1
        S3[1,1] = -1
        K = np.kron(N_gamma_0, S0) + np.kron(N_gamma_1, S1) + np.kron(N_gamma_2, S2) + np.kron(N_gamma_3, S3)
        # A_i s are Hermitian
        R = cp.Variable((n_cut**2,n_cut**2), symmetric=True)
        # compute partial trace of R
        constraints = [R >> 0]
        for i1 in range(n_cut):
            for j1 in range(n_cut):
                cons = np.zeros([n_cut**2,n_cut**2])
                for k in range(n_cut):
                    cons[i1*n_cut+k,j1*n_cut+k] = 1
                constraints += [cp.trace(cons@R)==np.eye(n_cut)[i1,j1]]

                        
        prob = cp.Problem(
            cp.Maximize(cp.real(1/8*cp.trace(K@R))),
            constraints
        )

        print('---begin optimization---')
        prob.solve()
        print('---end optimization---')
        # result choi matrix 
        #choi = 1/2*cp.kron(A0,np.identity(2)) + cp.kron(A1, sigma1)+ cp.kron(A2, sigma2)+ cp.kron(A3, sigma3) 
        choi = None
        return prob.value, choi

if __name__ == '__main__':
    Delta = 0.481
    gamma = 0.1
    n_cut = 13
    gkp = GKP_ampDamp(gamma,Delta)

    print(gkp.optimize_Recovery_numberBasis(n_cut)[0])

