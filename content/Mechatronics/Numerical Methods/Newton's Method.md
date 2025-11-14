In the simplest case:
$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$
Where:
- $x_n$ is your current guess
- $x_{n+1}$ is your improved guess
- $f'(x_n)$ is the derivative at $x_n$

>[!error] Newton's method will get us to a root (where $f(x)=0$), or it could shoot off if it reaches a point where $f'(x)=0$ and we end up dividing by 0
# For Optimization
For optimization we are often trying to find a minimum, not the 0 mark. This is done by doing Newton's Method on the derivative
$$
x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}
$$
---
**EXAMPLE** Newton's method for [[NLNG Problem Statement]]. lol

This is looking at deriving Newton's Optimization in a [[00 - Multivariate Calculus Table of Contents|Multivariate way]]

Given we have the Optimization problem from [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]]
$$
J(\mathbf{x}) = \frac{1}{2}\mathbf{u}(\mathbf{x})^T \mathbf{u}(\mathbf{x})
$$
And we end up with the final goal of
$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmin}}\;J(\mathbf{x})
$$
We can first do a Taylor Series expansion about an operating point $\mathbf{x}_{op}$ and a tiny arbitrary movement $\delta \mathbf{x}$
$$
J(\mathbf{x}_{\text{op}} + \delta\mathbf{x}) \approx J(\mathbf{x}_{\text{op}}) + \underbrace{\left(\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)}_{\text{Jacobian}} \delta\mathbf{x} + \frac{1}{2}\delta\mathbf{x}^T \underbrace{\left(\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right)}_{\text{Hessian}} \delta\mathbf{x}
$$
Because we want to **optimize**, we want to move $\delta \mathbf{x}$ in such a way that we end up at a **local minima of the cost function**. Hence where 
$$
\frac{\partial J(\mathbf{x}_{\text{op}} + \delta\mathbf{x})}{\partial \delta\mathbf{x}} = 0
$$
This gives us
$$
\left(\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) + \delta\mathbf{x}^T \left(\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right) =0
$$
Which hence lets us define a rough "movement" to move our operation point such that we reach a minimum
$$
\Rightarrow \quad \left(\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right) \delta\mathbf{x}^* = -\left(\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T
$$
We use $\delta \mathbf{x}^{*}$ to update our operating point
$$
\mathbf{x}_{op}\leftarrow \mathbf{x}_{op}+\delta \mathbf{x}^{*}
$$
Until we feel that we've reached a good enough location ($\delta \mathbf{x}^{*} <thresh$)

**Things to note:**
1. It'll converge to a minima, but that could be a global minima or more likely a local minima
2. The rate of convergence is quadratic
3. Hessian needs to be computed, which is hard in practice