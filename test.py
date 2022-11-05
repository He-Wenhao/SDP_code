from constants import *
import numpy as np
I = np.eye(2)
A = np.kron(I, I) + np.kron(sigma1, sigma1)+ np.kron(sigma2.transpose(), sigma2)+ np.kron(sigma3, sigma3)
print(np.linalg.eig(A))