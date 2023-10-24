# SDP_code

In this repository, we implement SDP method to obtain the optimal quantum error correction recovery.

Our implementation has following features:

1. the SDP has only one input: the QEC matrix.

2. the choi matrix is restricted on the error subspace. As a result, the choi matrix is $l*d*d$ dimensional, where l is the number of Kraus operators taken into consideration, $d=2$ for qubit code.

Also, we provide transpose channel and Choi matrix as a near optimal guess.

In warm-start branch, we manage to implement a warm start technique, basically there are two warm-start ideas:
1. use near optimal choi matrix as warm start
2. when iterate through some intercal, such as 0 <= gamma <=0.1 with small gamma steps, use previous gamma value as warm-start for the latter one
