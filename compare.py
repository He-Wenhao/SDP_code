import N_gamma
import SDP_optimize
import matplotlib.pyplot as plt
import numpy as np

def test_Msqrt():
    trans_data=[]
    SDP_data = []
    Delta = 0.309
    for gamma in np.linspace(0,0.1,11):
        dimL=20
        cutoff = 5
        res = N_gamma.tranpose_fid(Delta, gamma, dimL, cutoff)
        trans_data.append(res.real)
        print('trans:',res)

        n_cut = 40
        i_cut = 10
        gkp = SDP_optimize.GKP_ampDamp(gamma,Delta)
        res = 1-gkp.optimize_Recovery_numberBasis(n_cut)[0]
        SDP_data.append(res)
        print('SDP:',res)

    print('trans_data',trans_data)
    print('SDP_data',SDP_data)

    gamma = trans_data[1:]
    SDP = SDP_data[1:]
    trans = trans[1:]
    plt.plot(gamma, trans,label="trans")
    plt.plot(gamma, trans/3,label="1/3trans")
    plt.plot(gamma, SDP,label="SDP")
    plt.xlabel('gamma')
    plt.ylabel('infidelity')
    plt.title('Delta = 0.481')
    plt.legend(loc = 'lower right')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    test_Msqrt()