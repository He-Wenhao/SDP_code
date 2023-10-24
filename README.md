# SDP_code

In this repository, we implement SDP for optimal recovery. We consider entanglement fidelity

$$F_{\mathcal{N}_\gamma, \mathcal{R}}=\left\langle\Psi\left|\left(\mathcal{R} \circ \mathcal{N}_\gamma\right) \otimes \mathcal{I}(|\Psi\rangle\langle\Psi|)\right| \Psi\right\rangle.$$

SDP gives the optimal recodery
\begin{align}
F_{\mathcal{N}_\gamma, \mathcal{R}_{\mathrm{op}}}=\max _{\mathcal{R}}\left\langle\Psi\left|\left(\mathcal{R} \circ \mathcal{N}_\gamma\right) \otimes \mathcal{I}(|\Psi\rangle\langle\Psi|)\right| \Psi\right\rangle
\end{align}

Our implementation has following features:

1. the SDP has only one input: the QEC matrix.

2. the choi matrix is restricted on the error subspace. As a result, the choi matrix is $l*d*d$ dimensional, where l is the number of Kraus operators taken into consideration, $d=2$ for qubit code.

Also, we provide the results of transpose channel. 
\begin{align}
\mathcal{R}_T(\cdot) \equiv \sum_i P E_i^{\dagger} \mathcal{N}_\gamma(P)^{-1 / 2}(\cdot) \mathcal{N}_\gamma(P)^{-1 / 2} E_i P
\end{align}
Tranpose fidelity can serve as an approximation to the optimal fidelity, and transpose channel(in choi matrix form) serves as a good guess to optimized channel.

In warm-start branch, we manage to implement a warm start technique, basically there are two warm-start ideas:
1. use near optimal choi matrix as warm start
2. when iterate through some intercal, such as 0 <= gamma <=0.1 with small gamma steps, use previous gamma value as warm-start for the latter one
