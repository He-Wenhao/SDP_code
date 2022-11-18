from GKP_ampDamp import *
from constants import *
import qutip as qt
from functools import reduce
from scipy.optimize import minimize
import random
import warnings
import time
warnings.simplefilter('error', UserWarning)

a = qt.destroy


class sharpen_trim(GKP_nBasis):
    def __init__(self, Delta, gamma, n_cutoff, sum_cutoff,block_cnt):
        super(sharpen_trim, self).__init__(Delta, gamma, n_cutoff, sum_cutoff)
        self.para_cnt = 5    # parameter number in one block
        self.block_cnt = block_cnt
        self.paras = np.zeros(self.block_cnt*self.para_cnt,dtype=float)
        #self.K = self._reshape(self._K_nbasis().trans()).trans()
        nBasis = GKP_nBasis(self.Delta, self.gamma, self.n_cutoff, sum_cutoff = 5)    # for computing GKP
        self.orthBasis = nBasis.get_othNor_basis()

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
    # result on H_{ancillia} \otimes H_{code}
    def _block_U(self,para,der_ind):
        assert len(para) == self.para_cnt
        assert (der_ind in range(len(para))) or der_ind == None
        der = [(i==der_ind) for i in range(len(para))]
        CD = lambda i,t : self._controlD(para[i],t,der[i])
        R = lambda i,t: self._R(para[i],t,der[i])
        #return CD(0,'real')*CD(1,'imag')*R(2,'X')*R(3,'Y')*R(4,'Z')*CD(5,'real')*CD(6,'imag')
        return CD(0,'real')*R(1,'X')*CD(2,'imag')*R(3,'X')*CD(4,'real')

    # amplitude damping channel, with channel purification
    # result on  H_{code} \otimes H_{env}
    def _amp_damp(self):
        theta = np.arcsin(sqrt(self.gamma))
        a = qt.destroy(self.n_cutoff)
        b = qt.destroy(self.n_cutoff)
        iH = 1.j*theta*(qt.tensor(a,b.dag())+qt.tensor(a.dag(),b))
        return iH.expm()

    # prepare initial state
    # H_{puri} is the qubit for purification
    # result on H_{puri} \otimes H_{ancillia} \otimes H_{code} \otimes H_{env}
    def _init_state(self):
        H_puri = [qt.Qobj([[1.],[0.]]),qt.Qobj([[0.],[1.]])]
        phi_ancillia = 1/sqrt(2)*qt.Qobj([[1.],[1.]])
        phi_GKP = [qt.Qobj(self.orthBasis[0].transpose()),qt.Qobj(self.orthBasis[1].transpose())]
        phi_env = qt.basis(self.n_cutoff,0)
        return 1/sqrt(2)*(qt.tensor(H_puri[0],phi_ancillia,phi_GKP[0],phi_env)+qt.tensor(H_puri[1],phi_ancillia,phi_GKP[1],phi_env))

    

    

    # compute fidelity
    def fidelity(self,paras):
        myparas = np.array(paras).reshape((self.block_cnt,self.para_cnt))
        self.recovers = [self._block_U(myparas[i], None) for i in range(self.block_cnt)]
        state = self._init_state()
        Damp = qt.tensor(qt.qeye(2),qt.qeye(2),self._amp_damp())
        Recover = qt.tensor(qt.qeye(2),reduce(lambda x, y: x*y,self.recovers),qt.qeye(self.n_cutoff))
        return abs((state.dag()*Recover*Damp*state).tr())**2

    
    def optimize(self):
        res = minimize(lambda x:1-self.fidelity(x),self.paras,options={'disp': True})
        print(res)
    


if __name__ == '__main__':
    Delta = 0.309
    gamma = 0.1
    n_cutoff = 40
    sum_cutoff = 5
    block_cnt = 1
    epsilon = 0.001
    l = 2*sqrt(pi)
    st = sharpen_trim(Delta, gamma, n_cutoff, sum_cutoff, block_cnt)
    t = time.time()

    st.paras = [epsilon,pi/2,1.,1.,epsilon]
    print(st.fidelity(st.paras))
    t1 = time.time()
    print('time=',t1-t)
    st.paras = [epsilon,pi/2,-l,-pi/2,epsilon]
    print(st.fidelity(st.paras))
    t2 = time.time()
    print('time=',t2-t1)