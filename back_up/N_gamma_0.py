# this is representation with representation (7.8) in Performance and structure
# seems to be problematic



import numpy as np
import math

# compute N_\gamma(\ket{\mu}\ket{\nu}) , output a n_cut dimensional matrix
# n_cut is a cut off for phonon number
def N_gamma_normal_basis(mu, nu, Delta, gamma, n_cut, L_range):
    #L_range is lattice range
    #generate lattice points for mu and nu
    L_mu = [np.sqrt(np.pi/2)*((2*n1 + mu)+n2*1.j) 
            for n1 in range(-int(L_range/2),int(L_range/2)) 
            for n2 in range(-int(L_range),int(L_range))
        ]
    L_nu = [np.sqrt(np.pi/2)*((2*n1 + nu)+n2*1.j) 
            for n1 in range(-int(L_range/2),int(L_range/2)) 
            for n2 in range(-int(L_range),int(L_range))
        ]
    # compute matrix element
    result = np.zeros([n_cut,n_cut],dtype=complex)
    for i in range(n_cut):
        for j in range(n_cut):
            result[i,j] = sum([ 
                            np.exp(-(Delta**2+1/2)*(abs(a)**2+abs(ap)**2)) * np.exp(-1.j*(a.real*a.imag - ap.real*ap.imag)) * np.exp(gamma*a*ap.conjugate()) * a**i * (ap.conjugate())**j / np.sqrt(float(math.factorial(i)*math.factorial(j)))
                            for a in L_mu for ap in L_nu
                        ])
    return result

# use Pauli operator as input
def N_gamma_pauli_basis(pauli, Delta, gamma, n_cut, L_range):
    if pauli == 'I':
        return N_gamma_normal_basis(0,0, Delta, gamma, n_cut, L_range) + N_gamma_normal_basis(1,1, Delta, gamma, n_cut, L_range)
    elif pauli == 'X':
        return N_gamma_normal_basis(0,1, Delta, gamma, n_cut, L_range) + N_gamma_normal_basis(1,0, Delta, gamma, n_cut, L_range)
    elif pauli == 'Y':
        return -1.j*N_gamma_normal_basis(0,1, Delta, gamma, n_cut, L_range) + 1.j*N_gamma_normal_basis(1,0, Delta, gamma, n_cut, L_range)
    elif pauli == 'Z':
        return N_gamma_normal_basis(0,0, Delta, gamma, n_cut, L_range) - N_gamma_normal_basis(1,1, Delta, gamma, n_cut, L_range)
    else:
        raise TypeError('pauli operator type not found')

# general form, invoke this function recommended
def N_gamma(input,Delta,gamma,n_cut, L_range):
    if type(input) == str:
        return N_gamma_pauli_basis(input, Delta, gamma, n_cut, L_range)
    elif input in [(i,j) for i in range(2) for j in range(2)]:
        return N_gamma_normal_basis(input[0], input[1], Delta, gamma, n_cut, L_range)
    else:
        raise TypeError('input type error')


# just for test
if __name__ == '__main__':
    n_ave = 2
    Delta = 1/np.sqrt(2*n_ave + 1)
    gamma = 0.1
    n_cut = 10
    L_range = 7
    res = N_gamma('I', Delta, gamma, n_cut, L_range)
    print(np.trace(res))
    