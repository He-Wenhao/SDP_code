import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mpl_toolkits.mplot3d
from matplotlib import cm
df = pd.read_excel('data.xlsx',sheet_name='Sheet1',header = None)

n_Delta0 = np.array(df.iloc[1:,1])
gamma0 = np.array(df.iloc[1:,2])
infid_M0 = np.array(df.iloc[1:,3])
print('---')
n_Delta = np.array(list(set(n_Delta0)))
gamma = np.array(list(set(gamma0)))
infid_dict = {(n_Delta0[i],gamma0[i]):infid_M0[i] for i in range(len(n_Delta0))}
infid_M = np.array([[(infid_dict[(i,j)]) for i in n_Delta] for j in gamma])
n_Delta,gamma = np.meshgrid(n_Delta,gamma)
print(n_Delta)
#print(V2)


fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.plot_surface(gamma,n_Delta,infid_M,cmap=cm.ocean)
plt.show()

'''
m,b = np.polyfit(R, V2, 1)
print(m,b)

plt.scatter(R,V2,marker = 'x',label = 'data points',color = 'r')
x = np.linspace(R[0], R[-1], 1000)
plt.plot( x, m*x+b, label = 'linear fit')
plt.xlabel('R/$\Omega$')
plt.ylabel('$V_{rms}^2$/$V^2Hz^{-1}$')

plt.legend(loc = 'upper left')
plt.show()
'''