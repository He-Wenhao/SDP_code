import numpy as np
import cvxpy as cp
import scipy
from scipy import sparse
import scs

# processing QEC matrix
# only one input: QEC matrix (with orthornormal code basis)
# output: transpose fidelity, SDP fidelity
# Added functionality for direct use of SCS package

# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


class process_QECmat:
    def __init__(self,QECmat,dimL,d,scs_init=True):
        self.QECmat = QECmat
        self.dimL = dimL    # number of kraus operator taking into consideration
        self.d = d # d means qudit, 2 for qubit
        self.dimQECmat = d*dimL
        assert self.QECmat.shape == (self.dimQECmat,self.dimQECmat)
        self.scs_solver = None
        self.transchoi_x = None
        self.transchoi_y = None
        self.transchoi_s = None
        if scs_init:
            self.update_SCSprob(fresh_start=True)
        
    # compute transpose fidelity with equation 
    def transpose_infid_M(self):
        dimL = self.dimL
        Msqrt = scipy.linalg.sqrtm(self.QECmat)
        ptrMsqrt = Msqrt.reshape([dimL,2,dimL,2])
        ptrMsqrt = np.matrix([[ptrMsqrt[i,0,j,0]+ptrMsqrt[i,1,j,1] for j in range(dimL)] for i in range(dimL)])
        fid = (1/self.d**2)*np.trace(ptrMsqrt@ptrMsqrt.transpose())
        return 1-fid

    def transpose_choi(self):
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
        prob = cp.Problem((1/d**2)*cp.Maximize(cp.trace(cp.transpose(matK)@R)),constraints)
        return prob
    # generate choi matrix for transpose channel
    
    
    def update_SCSprob(self, fresh_start = False,tol=1.0e-6):
        dimL = self.dimL
        d = self.d
        # the SDP problem is
        # max_R Tr 1/4 + 1/4 \sum_{i=x,y,z} A_i N_\gamma (\sigma_i)
        # R = 1/2 I_n \otimes I_code + \sum_{i=x,y,z} A_i \otimes \sigma_i >= 0
        #compute N_gamma_othNor_pauli(sigma_i)
        
        if (fresh_start or (self.scs_solver == None)):
            A = []
            b = []
            # compute partial trace of R
            identMat = np.eye(dimL*d)
            for i1 in range(dimL):
                for j1 in range(dimL):
                    for mu1 in range(d):
                        for nu1 in range(d):
                            cons = np.zeros([dimL*d*d,dimL*d*d])
                            for k in range(self.d):
                                cons[k*dimL*d+i1*d+mu1,k*dimL*d+j1*d+nu1] = 1
                            A.append(vec(cons))
                            b.append(identMat[i1*d+mu1,j1*d+nu1])
                            #constraints += [cp.trace(cp.transpose(cons)@R)==identMat[i1*d+mu1,j1*d+nu1]]
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
            self.c = vec(-matK)/(d**2)
            self.b = np.array(b + [0]*len(self.c))
            self.A = sparse.vstack([sparse.csc_array(np.array(A)),
                                   -sparse.identity(len(self.c),format="csc")],
                                    format="csc")
            self.transchoi_x = vec(self.transpose_choi()) #Primal for transpose recovery
            self.transchoi_s = self.b - self.A@self.transchoi_x
            self.transchoi_y = sparse.linalg.lsqr(sparse.vstack([self.A.T,self.transchoi_s.T]), np.append(-self.c,[0]))[0] #Dual for transpose recovery
            data = dict(A=self.A,b=self.b,c=self.c)
            cone = dict(z=len(self.b)-len(self.c),s=d*dimL*d)
            self.scs_solver = scs.SCS(data=data,cone=cone,verbose=True,eps_abs=tol, eps_rel=tol)
        
        else:
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
            c_new = vec(-matK)/(d**2)
            self.c = c_new
            self.transchoi_y = sparse.linalg.lsqr(sparse.vstack([self.A.T,self.transchoi_s.T]), np.append(-self.c,[0]))[0]
            self.scs_solver.update(c=c_new)