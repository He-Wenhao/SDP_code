# %%
import os
import sys
sys.path.append('../..')
from GKP_ampDamp import *
from constants import *
import numpy as np

# %%
Delta = 0.05
print('Delta =',Delta)
n_D = comp_n_Delta(Delta)
print('n_Delta =',n_D)
gamma_lst = [0.01]
print('gamma list:',list(gamma_lst))


# %% [markdown]
# compute transpose infidelity directly

# %%

dimL=80
M_cutoff = 5
m_cutoff = 20
infid_M_lst = []
for gamma in gamma_lst:
    ElmuBasis = GKP_ElmuBasis(Delta = Delta, gamma = gamma, m_sum_cutoff=m_cutoff,M_sum_cutoff=M_cutoff,l_cut=dimL)
    # check
    if 1:
        ck = check_basis(ElmuBasis = ElmuBasis,nBasis = None)
        ck.trM()
    infid_M = ElmuBasis.transpose_infid_M()
    print(str(infid_M))
    infid_M_lst.append(infid_M)
print('infid_M_lst =',infid_M_lst)



