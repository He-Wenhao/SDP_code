'''
from constants import *
import numpy as np
I = np.eye(2)
A = np.kron(I, I) + np.kron(sigma1, sigma1)+ np.kron(sigma2.transpose(), sigma2)+ np.kron(sigma3, sigma3)
print(np.linalg.eig(A))
'''
class a:
    def __init__(self):
        self.A = 0

a1 = a()
print(type(a1))
print(type(a1) == a)