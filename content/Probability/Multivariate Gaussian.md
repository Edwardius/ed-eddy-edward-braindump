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

# Linear Change in a Gaussian
We have
$$
\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu_{x}}, \Sigma_{xx})=\frac{1}{(2\pi)^{d/2}\Sigma^{1/2}}\exp\left( -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) \right)
$$
It can be shown that if we have a linear transformation
$$
\mathbf{y}=\mathbf{G}\mathbf{x}
$$
The resultant Gaussian is
$$
y\sim \mathcal{N}(\mu_{y},\Sigma_{yy})=\mathcal{N}(\mathbf{G}\mu_{x},\mathbf{G}\Sigma_{xx}\mathbf{G}^T)
$$

# Non-Linear Operations on Gaussians

>[!info] There isn't a nice way of doing something like with with non-linear functions, so we Linearize by approximating a [[Jacobian]] near the point of operation

given a non-linear function
$$
\mathbf{y}=g(\mathbf{x})+v \;\; v\sim\mathcal{N}(\mathbf{0}, \mathbf{R})
$$
where $\mathbf{R}$ is the measurement noise covariance
we linearize like
$$
g(\mathbf{x}) \simeq \mu_{y}+\mathbf{G}(\mathbf{x}-\boldsymbol{\mu}_{x})
$$
$$
\mathbf{G}=\frac{ \partial g(\mathbf{x}) }{ \partial \mathbf{x} }|_{\mathbf{x}=\mu_{x}} 
$$
$$
\mu_{y}=g(\mu_{x})
$$
It can be shown with some work that the resultant
$$
\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{y},\boldsymbol{\Sigma}_{yy})=\mathcal{N}(g(\boldsymbol{\mu}_{x}),\mathbf{R}+\mathbf{G}\Sigma_{xx}\mathbf{G}^T)
$$
# Normalized Product of Gaussians
**The normalized product of N gaussians is also a gaussian.** Without normalization, we still end up with gaussian shape, just scaled by a constant.
$$
\exp\left( -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) \right)=\eta \prod ^{K}_{k=1}\exp\left( -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu_{k}})^{T}\Sigma_{k}^{-1}(\mathbf{x}-\boldsymbol{\mu_{k}}) \right)
$$
$$
\Sigma^{-1}=\sum ^{K}_{k=1}\Sigma^{-1}_{k}
$$
$$
\Sigma^{-1}\mu=\sum ^{K}_{k=1}\Sigma^{-1}_{k}\mu_{k}
$$
