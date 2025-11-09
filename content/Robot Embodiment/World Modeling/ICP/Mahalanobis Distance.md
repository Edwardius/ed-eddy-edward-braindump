#stateEstimation
Measures the "distance" a point is from a distribution. It measures the number of standard deviations a point is from a distribution.

In component form:
$$
D^{2}_{M}=\sum ^{p}_{i=1}\sum ^{p}_{j=1}(x_{i}-\mu_{i})\Sigma_{ij}^{-1}(x_{j}-\mu_{j})
$$
In matrix form:
$$
D^{2}_{M}(\mathbf{x})=(\mathbf{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

Note, this looks like the inside of the exponential of a [[Multivariate Gaussian]]! And it is!

$D_{M}^{2}$ measures the **number of standard deviations** a point is from the distribution.

#worldModeling