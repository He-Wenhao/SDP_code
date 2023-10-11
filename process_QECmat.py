import numpy as np
import cvxpy as cp
import scipy

# processing QEC matrix
# only one input: QEC matrix (with orthornormal code basis)
# output: transpose fidelity, SDP fidelity

class process_QECmat:
    def __init__(self,QECmat,dimL,d):
        self.QECmat = QECmat
        self.dimL = dimL    # number of kraus operator taking into consideration
        self.d = d # d means qudit, 2 for qubit
        self.dimQECmat = d*dimL
        assert self.QECmat.shape == (self.dimQECmat,self.dimQECmat)
        
    # compute transpose fidelity with equation 
    def transpose_infid_M(self):
        dimL = self.dimL
        Msqrt = scipy.linalg.sqrtm(self.QECmat)
        ptrMsqrt = Msqrt.reshape([dimL,2,dimL,2])
        ptrMsqrt = np.matrix([[ptrMsqrt[i,0,j,0]+ptrMsqrt[i,1,j,1] for j in range(dimL)] for i in range(dimL)])
        fid = (1/self.d**2)*np.trace(ptrMsqrt@ptrMsqrt.transpose())
        return 1-fid

    def tranpose_choi(self):
        dimL = self.dimL
        d = self.d
        trans_choi = np.matrix(np.zeros([dimL*d*d,dimL*d*d]))
        for i in range(dimL):
            for mu in range(d):
                for mup in range(d):
                    trans_choi[mu*d*dimL+i*d+mu,mup*d*dimL+i*d+mup] = 1
        return trans_choi
    
    
    # find an optimized recovery with SDP
    # use phonon number basis
    def SDP_set_prob(self):
        dimL = self.dimL
        d = self.d
        # the SDP problem is
        # max_R Tr 1/4 + 1/4 \sum_{i=x,y,z} A_i N_\gamma (\sigma_i)
        # R = 1/2 I_n \otimes I_code + \sum_{i=x,y,z} A_i \otimes \sigma_i >= 0
        #compute N_gamma_othNor_pauli(sigma_i)

        R = cp.Variable((d*dimL*d,d*dimL*d), symmetric=True)

        # compute partial trace of R
        constraints = [R >> 0]
        identMat = np.eye(dimL*d)
        for i1 in range(dimL):
            for j1 in range(dimL):
                for mu1 in range(d):
                    for nu1 in range(d):
                        cons = np.zeros([dimL*d*d,dimL*d*d])
                        for k in range(self.d):
                            cons[k*dimL*d+i1*d+mu1,k*dimL*d+j1*d+nu1] = 1
                        constraints += [cp.trace(cp.transpose(cons)@R)==identMat[i1*d+mu1,j1*d+nu1]]
        _Msqrt = np.matrix(scipy.linalg.sqrtm(self.QECmat))
        Msqrt = np.matrix(np.zeros([d*dimL*d,d*dimL*d]))
        for i in range(dimL*d):
            for j in range(dimL*d):
                for mu in range(d):
                    Msqrt[mu*dimL*d+i,mu*dimL*d+j] = _Msqrt[i,j]
        
        matK = np.matrix(np.zeros([d*dimL*d,d*dimL*d]))
        for i in range(dimL):
            for mu in range(d):
                for mup in range(d):
                    matK[mu*dimL*d+i*d+mu,mup*dimL*d+i*d+mup] = 1
                    
        # solve SDP
        matK = Msqrt.dot(matK).dot(Msqrt)                
        prob = cp.Problem(
            (1/d**2)*cp.Maximize(cp.trace(cp.transpose(matK)@R)),
            constraints
        )
        return prob
    
    # generate choi matrix for transpose channel
