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
        # A_i s are Hermitian
        A0 = cp.Variable((n_cut,n_cut), hermitian=True)
        A1 = cp.Variable((n_cut,n_cut), hermitian=True)
        A2 = cp.Variable((n_cut,n_cut), hermitian=True)
        A3 = cp.Variable((n_cut,n_cut), hermitian=True)

        prob = cp.Problem(
            cp.Maximize(cp.real(1/4*cp.trace(A0 @ N_gamma_0)+1/4*cp.trace(A1 @ N_gamma_1)+1/4*cp.trace(A2 @ N_gamma_2)+1/4*cp.trace(A3 @ N_gamma_3))),
            [
                1/2*cp.kron(A0,np.identity(2)) + cp.kron(A1, sigma1)+ cp.kron(A2, sigma2)+ cp.kron(A3, sigma3) >> 0,
                np.identity(n_cut) - A0 >> 0,
                A0 >> 0
            ]
        )

        print('---begin optimization---')
        prob.solve()
        print('---end optimization---')
        # result choi matrix 
        choi = 1/2*cp.kron(A0,np.identity(2)) + cp.kron(A1, sigma1)+ cp.kron(A2, sigma2)+ cp.kron(A3, sigma3) 
        return prob.value, choi

if __name__ == '__main__':
    Delta = 0.481
    gamma = 0.0
    n_cut = 70
    gkp = GKP_ampDamp(gamma,Delta)

    print(gkp.optimize_Recovery_numberBasis(n_cut)[0])

