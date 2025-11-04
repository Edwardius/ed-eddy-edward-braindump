#probability

For review of multivariate calculus, see [[00 - Multivariate Calculus Table of Contents]].

# Normal Distribution (Gaussian) in 1D
Given by
$$
p(x) \sim \mathcal{N}(\mu,\sigma^{2}) = \frac{1}{\sigma\sqrt{ 2\pi }} \exp\left( -\frac{1}{2}\left( \frac{x-\mu}{\sigma} \right)^{2}\right)
$$

![[Pasted image 20251103205534.png]]
- The higher the $\sigma$ the "wider the distribution"
- $\sigma$ is the standard deviation, $\sigma^{2}$ is the variance

# Multivariate Gaussian
Given by
$$
p(\mathbf{x})\sim\mathcal{N}(\boldsymbol{\mu}, \Sigma)=\frac{1}{(2\pi)^{d/2}\Sigma^{1/2}}\exp\left( -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) \right)
$$

![[Pasted image 20251104104439.png]]
Where $\Sigma$ is called the **covariance matrix**, it consists of the following:
- **Diagonal $\Sigma$** variance in each dimension
- **Off Diagonal $\Sigma$** [[Covariance | covariance]] between two dimensions
$$
\Sigma=\begin{pmatrix}Var(X) & Cov(X,Y) \\
Cov(Y,X) & Var(Y)\end{pmatrix}
$$
>[!info] A covariance matrix is, by definition, symmetric.

$$
Cov(Y,X)=E[(Y-\mu_{Y})(X-\mu_{X})]=E[(X-\mu_{X})(Y-\mu_{Y})]=Cov(X,Y)
$$