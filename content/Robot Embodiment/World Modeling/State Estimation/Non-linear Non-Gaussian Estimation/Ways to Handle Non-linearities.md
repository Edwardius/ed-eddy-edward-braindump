There are a number of methods to help us deal with [[NLNG Problem Statement]].

1. **Linearization** as shown with [[Extended Kalman Filter]]
2. [[Monte Carlo Method]] "brute force" method that randomly samples values in the input distribution, run them through the non-linearity, characterize the output distribution.
	1. Pretty dumb but its actually really accurate (law of large numbers says that you will get to the exact distribution as you reach a infinite number of samples)
	2. Used as an evaluation method
	3. Works with any PDF, not just Gaussians
3. [[Sigmapoint (Unscented) Transformation]] Its kinda seen as a compromise between full linearization and the monte carlo method

---
**EXAMPLE** Say we have a 1D non-linearity $f(x)=x^{2}$ and the prior density is $\mathcal{N}(\mu_{x},\sigma^{2}_{x})$

**Using Monte Carlo Method**
There is a closed form, exact answer to this solution, so we don't need to randomly sample.
$$
x_{i}=\mu_{x}+\delta x_{i},\;\; \delta x_{i}\sim\mathcal{N}(0,\sigma^{2}_{x})
$$
Transforming it through the non-linearity we get:
$$
y_{i}=(\mu_{x}+\delta x_{i})^{2}=\mu_{x}^{2}+2\mu_{x}\delta x_{i}+\delta x_{i}^{2}
$$
Grabbing the characteristics:
$$
\mu_{y}=E[y_{i}]=\mu_{x}^{2}+2\mu_{x}\underbrace{ E[\delta x_{i}] }_{ 0 }+\underbrace{ E[\delta x_{i}^{2}] }_{ \sigma_{x}^{2} }=\mu_{x}^{2}+\sigma^{2}_{x}
$$
$$
\sigma^{2}_{y}=E[(y_{i}-\mu_{y})^{2}]=2\mu_{x}^{2}\sigma_{x}^{2}+2\sigma_{x}^{4}
$$
In truth the resulting output density is not a Gaussian, but it can resultantly be approximated as one with $\mathcal{N}(\mu_{y},\sigma_{y}^{2})$

**Using Linearization**
$$
y_{i}=f(\mu_{x}+\delta x_{i})\simeq \underbrace{ f(\mu_{x}) }_{ \mu_{x^{2}} }+\underbrace{ \frac{ \partial f }{ \partial x }|_{\mu_{x}} }_{ 2\mu_{x} }\delta x_{i}=\mu_{x}^{2}+2\mu_{x}\delta x_{i}
$$
$$
\mu_{y}=E[y_{i}]=\mu_{x}^{2}+2\mu_{x}\underbrace{ E[\delta x_{i}] }_{ 0 }=\mu_{x}^{2}
$$
$$
\sigma_{x}^{2}=E[(y_{i}-\mu_{i}^{2})]=E[(2\mu_{x}\delta x_{i})^{2}]=4\mu_{x}^{2}\sigma_{x}^{2}
$$
Which as you can see already has some descrepancies with our Monte Carlo Method. **The linearized mean has a bias and the variance is too small**.

**Using Sigmapoint Transformation**
Its 1D  so we need $2L+1=3$ sigmapoints.
$$
x_{0}=\mu_{x}, \;\;x_{1}=\mu_{x}+\sqrt{ 1+\kappa }\sigma_{x}, \;\;x_{1}=\mu_{x}-\sqrt{ 1+\kappa }\sigma_{x}
$$
We can send these points through the nonlinearity.
$$
y_{o}=f(x_{o})=\mu_{x}^{2}
$$
$$
y_{1}=f(x_{1})=(\mu_{x}+\sqrt{ 1+\kappa }\sigma_{x})^{2}=\mu_{x}^{2}+2\mu_{x}\sqrt{ 1+\kappa }\sigma_{x}+(1+\kappa)\sigma_{x}^{2}
$$
$$
y_{2}=f(x_{2})=(\mu_{x}-\sqrt{ 1+\kappa }\sigma_{x})^{2}=\mu_{x}^{2}-2\mu_{x}\sqrt{ 1+\kappa }\sigma_{x}+(1+\kappa)\sigma_{x}^{2}
$$
Mean is given by 
$$
\mu_{y}=\frac{1}{1+\kappa}\left( \kappa y_{0}+\frac{1}{2}\sum ^{2}_{i=1}y_{i} \right)=\mu_{x}^{2}+\sigma_{x}^{2}
$$
Variance is given by
$$
\sigma_{y}^{2}=\frac{1}{1+\kappa}\left( \kappa(y_{0}-\mu_{y})^{2}+\frac{1}{2}\sum ^{2}_{i=1}(y_{i}-\mu_{y})^{2} \right)=4\mu_{x}^{2}\sigma_{x}^{2}+\kappa \sigma_{x}^{2}
$$
Which means that by having the user set $\kappa=2$ we end up with the correct mean and covariance of the output.

![[Pasted image 20251113144642.png]]


