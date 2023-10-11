# %% [markdown]
# In this ipynb file, we implement SDP for optimal recovery. Our implementation has following features:
# 
# 1. the SDP has only one input: the QEC matrix.
# 
# 2. the choi matrix is restricted on the error subspace. As a result, the choi matrix is l*d*d dimensional, where l is the number of Kraus operators taken into consideration, d=2 for qubit code.
# 
# Furthermore, we provide an near optimal solution to the SDP

# %%
import warnings
warnings.filterwarnings("error")
import numpy as np
import math
import mpmath
from scipy.special import hermite,  eval_genlaguerre, loggamma
from functools import lru_cache
import cvxpy as cp 
import scipy


sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
sqrt = np.sqrt
pi = np.pi
exp = np.exp
conj = np.conj
ln = np.log

# %% [markdown]
# Construct QEC matrix. Here we use Gaussian envelope finite energy GKP code as an example, you can substitute it for other codes. The GKP matrix $M_{l,\mu,l',\mu'}$ aligns as following:

# %%
from QECmat_GKP import QECmat_GKP

# %% [markdown]
# implement SDP process

# %%
from process_QECmat import process_QECmat


# %% [markdown]
# more test

# %%
for gamma in np.linspace(0,0.1,11):
    ####################
    # set parameters for GKP
    ####################
    print('--- gamma =',gamma)
    Delta = 0.481
    dimL=20
    cutoff = 5
    l_cut = 20
    m_sum_cutoff=20
    M_sum_cutoff=5
    
    n_cut = 40
    i_cut = 10
    eps = 1e-6
    
    ####################
    # calculate QEC matrix for GKP
    ####################
    M_GKP = QECmat_GKP(Delta = Delta, gamma = gamma, l_cut = l_cut, m_sum_cutoff = m_sum_cutoff, M_sum_cutoff = M_sum_cutoff).orth_M()
    
    ####################
    # calculate optimal/near optimal recovery
    ####################
    GKP_QECprocess = process_QECmat(QECmat = M_GKP,dimL = l_cut,d = 2)  # input GKP QEC matrix
    
    # calculate transpose fidelity result directly
    trans_infid = GKP_QECprocess.transpose_infid_M()
    print('transpose infid direct:',trans_infid)

    # calculate transpose choi matrix, and test consistency
    trans_choi = GKP_QECprocess.tranpose_choi()
    prob0 = GKP_QECprocess.SDP_set_prob()  # setup SDP problem
    prob0.variables()[0].save_value(trans_choi)# feed choi into the variable
    print('transpose infid by choi:',1-prob0.objective.value)
    

    # do SDP optimization
    prob1 = GKP_QECprocess.SDP_set_prob()  # setup SDP problem
    prob1.solve(eps=eps,verbose = False)   # solve SDP
    print('infid by SDP:',1-prob1.value,'iter:',prob1._solver_stats.num_iters)


