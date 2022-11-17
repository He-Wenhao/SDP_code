from GKP_ampDamp import *
from constants import *
import qutip as qt
from functools import reduce
from scipy.optimize import minimize
import random
import warnings
warnings.simplefilter('error', UserWarning)

a = qt.destroy

class sharpen_trim(GKP_nBasis):
    def __init__(self, Delta, gamma, n_cutoff, sum_cutoff,block_cnt):
        super(sharpen_trim, self).__init__(Delta, gamma, n_cutoff, sum_cutoff)
        self.para_cnt = 5    # parameter number in one block
        self.block_cnt = block_cnt
        self.paras = np.zeros(self.block_cnt*self.para_cnt,dtype=float)
        self.K = self._reshape(self._K_nbasis().trans()).trans()

    def _D(self,alp,type_alp,der = False):
        assert type_alp in ['real','imag']
        a = qt.destroy(self.n_cutoff)
        if type_alp == 'real':
            iH = alp*a.dag()-alp*a
        elif type_alp == 'imag':
            iH = 1.j*(alp*a.dag()+alp*a)
        if not der:
            return iH.expm()
        else:
            return iH*iH.expm()

    def _controlD(self,alp,type_alp,der = False):
        if not der:
            return qt.tensor(qt.Qobj([[1,0],[0,0]]),qt.qeye(self.n_cutoff)) + qt.tensor(qt.Qobj([[0,0],[0,1]]),self._D(alp,type_alp,der))
        else:
            return qt.tensor(qt.Qobj([[0,0],[0,1]]),self._D(alp,type_alp,der))

    def _R(self,theta,type_theta,der=False):
        assert type_theta in ['X','Y','Z']
        if type_theta == 'X':
            sigma = qt.sigmax()
        elif type_theta == 'Y':
            sigma = qt.sigmay()
        elif type_theta == 'Z':
            sigma = qt.sigmaz()
        iH = 1.j*theta*sigma
        if not der:
            res = iH.expm()
        else:
            res = iH*iH.expm()
        return qt.tensor(res,qt.qeye(self.n_cutoff))

    # unitary for one block
    def _block_U(self,para,der_ind):
        assert len(para) == self.para_cnt
        assert (der_ind in range(len(para))) or der_ind == None
        der = [(i==der_ind) for i in range(len(para))]
        CD = lambda i,t : self._controlD(para[i],t,der[i])
        R = lambda i,t: self._R(para[i],t,der[i])
        #return CD(0,'real')*CD(1,'imag')*R(2,'X')*R(3,'Y')*R(4,'Z')*CD(5,'real')*CD(6,'imag')
        return CD(0,'real')*R(1,'X')*CD(2,'imag')*R(3,'X')*CD(4,'real')

    # choi matrix for one block
    def _block_choi(self,para,der_ind):
        if der_ind == None:
            U = self._block_U(para,der_ind)
            Xeigen = 0.5*qt.Qobj([[1,-1],[-1,1]])    # |+> state
            ijBasis = lambda i,j:qt.basis(self.n_cutoff,i)*qt.basis(self.n_cutoff,j).dag()
            channelij = lambda i,j: (U*qt.tensor(Xeigen,ijBasis(i,j))*U.dag()).ptrace(1)
            return sum([qt.tensor(ijBasis(i,j),channelij(i,j)) for i in range(self.n_cutoff) for j in range(self.n_cutoff)])
        else:
            U = self._block_U(para,None)
            U_der = self._block_choi(para,der_ind)
            Xeigen = 0.5*qt.Qobj([[1,-1],[-1,1]])    # |+> state
            ijBasis = lambda i,j:qt.basis(self.n_cutoff,i)*qt.basis(self.n_cutoff,j).dag()
            channelij = lambda i,j: (U_der*qt.tensor(Xeigen,ijBasis(i,j))*U.dag()).ptrace(1) + (U*qt.tensor(Xeigen,ijBasis(i,j))*U_der.dag()).ptrace(1)
            return sum([qt.tensor(ijBasis(i,j),channelij(i,j)) for i in range(self.n_cutoff) for j in range(self.n_cutoff)])

    # reshape choi matrix
    # convert A_{ij,kl} into A_{ik,jl}
    def _reshape(self,A):
        b = np.matrix(A)
        res = b.copy()
        dims = A.dims
        iR,jR,kR,lR = dims[0][0],dims[0][1],dims[1][0],dims[1][1]
        for i in range(iR):
            for j in range(jR):
                for k in range(kR):
                    for l in range(lR):
                        res[i*jR+j,k*lR+l] = b[i*kR+k,j*lR+l]
        return qt.Qobj(res,dims = dims)

    # obtain k,but with n basis
    def _K_nbasis(self):
        K = self._K_forSDP()
        P = np.matrix(self.get_othNor_basis())
        P = np.kron(np.eye(self.n_cutoff), P)
        return qt.Qobj(P.transpose()@K@P,dims = [[self.n_cutoff,self.n_cutoff],[self.n_cutoff,self.n_cutoff]])

    # compute fidelity
    def fidelity(self,paras):
        myparas = np.array(paras).reshape((self.block_cnt,self.para_cnt))
        self.chois = [self._reshape(self._block_choi(myparas[i], None)) for i in range(self.block_cnt)]
        return reduce(lambda x, y: x*y,self.chois+[self.K]).tr()

    '''
    def optimize(self):
        res = minimize(lambda x:-self.fidelity(x),self.paras,method = 'BFGS')
        print(res)
    '''


if __name__ == '__main__':
    Delta = 0.309
    gamma = 0.1
    n_cutoff = 40
    sum_cutoff = 5
    block_cnt = 1
    epsilon = 0.001
    l = 2*sqrt(pi)
    st = sharpen_trim(Delta, gamma, n_cutoff, sum_cutoff, block_cnt)
    st.paras = [epsilon,pi/2,-l,-pi/2,epsilon]
    print(st.fidelity(st.paras))