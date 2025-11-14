A compromise between linearization techniques like the [[Extended Kalman Filter]] and [[Monte Carlo Method]].

1. A set of $2L+1$ *sigmapoints* $\mathbf{x}_{i}$ are compute from the input density, $\mathcal{N}(\boldsymbol{\mu}_{x},\boldsymbol{\Sigma}_{\times})$, according to
$$
\mathbf{L}\mathbf{L}^{T}=\boldsymbol{\Sigma}_{xx}
$$
$$
\mathbf{x}_{0}=\boldsymbol{\mu}_{x}
$$
$$
\mathbf{x}_{i}=\boldsymbol{\mu}_{x}+\sqrt{ L+\mathcal{K} }col_{i}\mathbf{L}
$$
$$
\mathbf{x}_{i+L}=\boldsymbol{\mu}_{x}-\sqrt{ L+\mathcal{K} }col_{i}\mathbf{L}
$$
You can convert the sigmapoints back to the distribution by
$$
\boldsymbol{\mu}_{x}=\sum ^{2L}_{i=0}\alpha_{i}\mathbf{x}_{i}
$$
$$
\boldsymbol{\Sigma}_{xx}=\sum ^{2L}_{i=0}\alpha_{i}(\mathbf{x}_{i}-\boldsymbol{\mu}_{i})(\mathbf{x}_{i}-\boldsymbol{\mu}_{x})^{T}
$$
$$
\alpha_{i}=\begin{cases}
\frac{\kappa}{L+\kappa}, & i=0 \\
\frac{1}{2} \frac{1}{L+\kappa} & otherwise
\end{cases}
$$
$\kappa$ here is a **user defined parameter.**

>[!error] $\kappa$ lets you define how far away the sigmapoints are from the mean. It affects the fourth moment of the distribution which is known as *kurtosis*

where $L=dim(\boldsymbol{\mu}_{x})$ is the dimensionality of the mean

![[Pasted image 20251113134354.png]]
2. For each *sigmapoint* , pass it through the nonlinearity
$$
\mathbf{y}_{i}=\mathbf{g}(\mathbf{x}_{i}), \;\;i=0\dots 2L
$$
3. The mean of the output density is computed as
$$
\boldsymbol{\mu}_{y}=\sum ^{2L}_{i=0}\alpha_{i}\mathbf{y}_{i}
$$
4. The covariance of the output density, $\boldsymbol{\Sigma}_{yy}$, is computed as
$$
\boldsymbol{\Sigma}_{yy}=\sum ^{2L}_{i=0}\alpha_{i}(\mathbf{y}_{i}-\boldsymbol{\mu}_{y})(\mathbf{y}_{i}-\boldsymbol{\mu}_{y})^{T}
$$
5. The output density $\mathcal{N}(\boldsymbol{\mu}_{y},\boldsymbol{\Sigma}_{yy})$ is returned

# Advantages
1. We can avoid finding the [[Jacobian]] of the non-linearity
2. only standard linear algebra operations are employed
3. The computation cost is similar to linearization (when a numerically derived Jacobian is used)
4. **There is no requirement that the nonlinearity be smooth and differentiable**

