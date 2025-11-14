Used to solve a **non-linear least squares** problems.
$$
\min_{\mathbf{x}} F(\mathbf{x}) = \min_{\mathbf{x}} \frac{1}{2}\sum_{i=1}^{n} r_i(\mathbf{x})^{2}\;\text{(component form)} = \min_{\mathbf{x}} \frac{1}{2}\|\mathbf{r}(\mathbf{x})\|^2\;\text{(matrix form)}
$$
$$
=\min_{\mathbf{x}} \frac{1}{2}\mathbf{r}(\mathbf{x})^{T}\mathbf{r}(\mathbf{x})
$$
where:
- $\mathbf{x} \in \mathbb{R}^m$ is the vector of parameters we want to optimize
- $\mathbf{r}(\mathbf{x}) = [r_1(\mathbf{x}), r_2(\mathbf{x}), \ldots, r_n(\mathbf{x})]^T \in \mathbb{R}^n$ is the residual vector
- Each $r(x_{i})$ is a non-linear function measuring the **error in the ith observation**
- 1/2 is for convenience when computing the derivative
# What constitutes a non-linear least squares problem?
Under gaussian assumptions, minimizing the non-linear least squares problem is equivalent to giving us the **maximum likelihood estimate**

# Key Assumption
Gauss-Newton assumes **that our initial guess is relatively near our optimum**.
At its core, Gauss-Newton is an extension of [[Newton's Method]] but attempting to handle the deriving the hessian problem.

From our example in [[Newton's Method]], if we try to actually derive the [[Jacobian]] and [[Hessian]], we get: 
$$
\text{Jacobian: }\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}} = \mathbf{u}(\mathbf{x}_{\text{op}})^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)
$$$$
\text{Hessian:  }\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}} = \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) + \cancelto{ 0 }{ \sum_{i=1}^{n} u_i(\mathbf{x}_{\text{op}}) \left(\frac{\partial^2 u_i(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right) }
$$
The Hessian here is tough to compute, especially when you need to consider the non-linearity.

>[!error] To get around this, we assume that we are already near the local minimum and thus our error $u_{i}(\mathbf{x}_{op})$ is small (ideally 0). So we just get rid of the hessian altogether and approximate the hessian.

$$
\text{Hessian:  }\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}} = \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) + \cancelto{ 0 }{ \sum_{i=1}^{n} u_i(\mathbf{x}_{\text{op}}) \left(\frac{\partial^2 u_i(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right) }
$$
$$
\text{Hessian:  }\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}} = \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)
$$
Expanding on this assumption, we lead towards the Gauss-Newton Method
$$
\Rightarrow \quad \left(\frac{\partial^2 J(\mathbf{x})}{\partial \mathbf{x} \partial \mathbf{x}^T}\bigg|_{\mathbf{x}_{\text{op}}}\right) \delta\mathbf{x}^* = -\left(\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T
$$
$$
\left(\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)\right) \delta\mathbf{x}^* = -\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \mathbf{u}(\mathbf{x}_{\text{op}})
$$
Which leads to the same derivation as in the **EXAMPLE**
# How to Gauss-Newton
At its core, we linearize the residuals, solve the linear least squares problem, update our estimate, rinse and repeat.

1. Initialize an initial guess $\mathbf{x}_0$ , following guesses are denoted as $\mathbf{x}_{k}$
2. Linearize the residuals (this is using first order [[Taylor Series]])
$$
r_i(\mathbf{x}_k + \Delta\mathbf{x}) \approx r_i(\mathbf{x}_k) + \nabla r_i(\mathbf{x}_k)^T \Delta\mathbf{x}
$$
Where $\nabla r_{i}$ is the gradient of the residual

In matrix (vector) form
$$
\mathbf{r}(\mathbf{x}_k + \Delta\mathbf{x}) \approx \mathbf{r}(\mathbf{x}_k) + \mathbf{J}_k \Delta\mathbf{x}
$$
Where $\mathbf{J}_{k}$ is the [[Jacobian]] of $r$ evaluated at $\mathbf{x}_{k}$
3. Formulate the linear least squares problem
$$
\min_{\Delta\mathbf{x}} \frac{1}{2}\|\mathbf{r}(\mathbf{x}_k + \Delta\mathbf{x})\|^2 \approx \min_{\Delta\mathbf{x}} \frac{1}{2}\|\mathbf{r}(\mathbf{x}_k) + \mathbf{J}_k \Delta\mathbf{x}\|^2
$$
4. Derive the normal equation
$$
f(\Delta\mathbf{x}) = \frac{1}{2}(\mathbf{r}_k + \mathbf{J}_k \Delta\mathbf{x})^T (\mathbf{r}_k + \mathbf{J}_k \Delta\mathbf{x})\;\;\text{where}\;\;r_{k}=r(\mathbf{x}_{k})
$$
$$
f(\Delta\mathbf{x}) = \frac{1}{2}\mathbf{r}_k^T\mathbf{r}_k + \mathbf{r}_k^T\mathbf{J}_k\Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T\mathbf{J}_k^T\mathbf{J}_k\Delta\mathbf{x}
$$
$$
\frac{\partial f}{\partial \Delta\mathbf{x}} = \mathbf{J}_k^T\mathbf{r}_k + \mathbf{J}_k^T\mathbf{J}_k\Delta\mathbf{x}
$$
$$
\mathbf{J}_k^T\mathbf{J}_k\Delta\mathbf{x} +\mathbf{J}_k^T\mathbf{r}_k= 0
$$
5. Solve for the Update
$$
\Delta\mathbf{x}_k = -(\mathbf{J}_k^T\mathbf{J}_k)^{-1}\mathbf{J}_k^T\mathbf{r}_k
$$
>[!info] we don't normally compute the inverse here, just solve for the linear equation instead.

$$
\mathbf{J}_k^T\mathbf{J}_k\Delta\mathbf{x}_k = -\mathbf{J}_k^T\mathbf{r}_k
$$
6. Update the parameters
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x}_k
$$
7. Check convergence by: checking if the step size is small, it the gradient is small, if there is a small relative change, max iterations, etc.

---
**EXAMPLE** in non-linear non-gaussian state estimation
Following [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] for a [[NLNG Problem Statement]]

We have the following cost function:
$$
J(\mathbf{x}) = \frac{1}{2}\mathbf{u}(\mathbf{x})^T \mathbf{u}(\mathbf{x})
$$
where, related to what has been said above
$$
\mathbf{u}(\mathbf{x})=\mathbf{r}(\mathbf{x})
$$
We **linearize $\mathbf{u}(\mathbf{x}_{op}+\delta \mathbf{x})$**
$$
\mathbf{u}(\mathbf{x}_{op}+\delta \mathbf{x})\simeq \mathbf{u}(\mathbf{x}_{op})+\left( \frac{ \partial \mathbf{u}(\mathbf{x}) }{ \partial \mathbf{x} } |_{\mathbf{x}_{op}}\right)\delta \mathbf{x}
$$
Substituting into $J(\mathbf{x}_{op}+\delta \mathbf{x})$
$$
J(\mathbf{x}_{op}+\delta \mathbf{x})\simeq \frac{1}{2} \left(\mathbf{u}(\mathbf{x}_{op})+\left( \frac{ \partial \mathbf{u}(\mathbf{x}) }{ \partial \mathbf{x} } |_{\mathbf{x}_{op}}\right)\delta \mathbf{x}\right)^{T}\left(\mathbf{u}(\mathbf{x}_{op})+\left( \frac{ \partial \mathbf{u}(\mathbf{x}) }{ \partial \mathbf{x} } |_{\mathbf{x}_{op}}\right)\delta \mathbf{x}\right)
$$
$$
\frac{\partial J(\mathbf{x}_{\text{op}} + \delta\mathbf{x})}{\partial \delta\mathbf{x}} = \left(\mathbf{u}(\mathbf{x}_{\text{op}}) + \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)\delta\mathbf{x}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) = 0
$$
$$
\Rightarrow \quad \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) \delta\mathbf{x}^* = -\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \mathbf{u}(\mathbf{x}_{\text{op}})
$$
 We solve for $\delta \mathbf{x}^{*}$ to update our operating point until we are happy
 $$
\mathbf{x}_{op}\leftarrow \mathbf{x}_{op}+\delta \mathbf{x}^{*}
$$
# Patches for Practicality
1. Adding a step size
 $$
\mathbf{x}_{op}\leftarrow \mathbf{x}_{op}+\alpha\;\delta \mathbf{x}^{*}
$$
	between 0 and 1, just lets us control our step size. Good to do a sweep of this
2. If our assumption on the hessian approximation is poor (can occur if we are far from the optimum (in SLAM that's when our initial guess is very bad)), we might not be able to converge very well.

	To help with this, we can use the [[Levenberg-Marquardt Modification]]