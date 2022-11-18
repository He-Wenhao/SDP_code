import matplotlib.pyplot as plt
import numpy as np
gamma = np.linspace(0,0.1,11)
trans = np.array([1.3322676295501878e-15, 0.0007032768163517389, 0.001578879275889955, 0.002638626255128451, 0.003894026091168401, 0.005356332498326677, 0.007036571652055845, 0.008945546653714165, 0.011093824705171396, 0.013491711403485374, 0.016149215699938435])
trans_1st = np.array([])
trans_2st = np.array([])
SDP = np.array([-7.455298149938727e-07, 0.0006865188110174447, 0.0015222985893250662, 0.0024676647556851616, 0.003556396107759374, 0.004830573946393302, 0.00626432391744447, 0.00788656772789631, 0.009688146949182075, 0.011724248503141688, 0.013922140516446246])
gamma = gamma[1:]
SDP = SDP[1:]
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