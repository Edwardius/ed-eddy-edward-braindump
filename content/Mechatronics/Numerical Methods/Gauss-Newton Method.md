Used to solve a **non-linear least squares** problems.
$$
\min_{\mathbf{x}} F(\mathbf{x}) = \min_{\mathbf{x}} \frac{1}{2}\sum_{i=1}^{n} r_i(\mathbf{x})^{2}\;\text{(component form)} = \min_{\mathbf{x}} \frac{1}{2}\|\mathbf{r}(\mathbf{x})\|^2\;\text{(matrix form)}
$$
where:
- $\mathbf{x} \in \mathbb{R}^m$ is the vector of parameters we want to optimize
- $\mathbf{r}(\mathbf{x}) = [r_1(\mathbf{x}), r_2(\mathbf{x}), \ldots, r_n(\mathbf{x})]^T \in \mathbb{R}^n$ is the residual vector
- Each $r(x_{i})$ is a non-linear function measuring the **error in the ith observation**
- 1/2 is for convenience when computing the derivative
# What constitutes a non-linear least squares problem?
Under gaussian assumptions, minimizing the non-linear least squares problem is equivalent to giving us the **maximum likelihood estimate**

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