From [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] we have the loss function
$$
J(\mathbf{x})=\frac{1}{2}\sum ^{N}_{i=1}\mathbf{e}_{i}(\mathbf{x})^{T}\mathbf{W}^{-1}_{i}(\mathbf{e}_{i}(\mathbf{x}))
$$
The gradient of which (which is used for [[Gauss-Newton Method]] is given by
$$
\frac{ \partial J(\mathbf{x}) }{ \partial \mathbf{x} } =\sum_{i=1}^{N} \mathbf{e}_{i}(\mathbf{x})^{T}\mathbf{W}_{i}^{-1}\frac{ \partial \mathbf{e}_{i}(\mathbf{x}) }{ \partial \mathbf{x} } 
$$
Because the cost function is quadratic, **our cost explodes when outliers cause major errors**. To handle this, there are a series of *wrappers* that we can use to limit the effects of robot loss functions.
$$
J'(\mathbf{x})=\sum_{i=1}^{N} \alpha_{i}\rho(u_{i}(\mathbf{x})) \quad u_{i}(\mathbf{x})=\sqrt{ \mathbf{e}_{i}(\mathbf{x})^{T}\mathbf{W}^{-1}_{i}(\mathbf{e}_{i}(\mathbf{x})) }
$$
where $\alpha$ is a scalar weight you can define, $\rho$ is some **non-linear cost function** (the wrapper).

Some possible cost functions are
$$
\underbrace{ \rho(u)=\frac{1}{2}\ln(1+u^{2}) }_{ Cauchy }\quad \underbrace{ \rho(u)=\frac{1}{2} \frac{u^{2}}{1+u^{2}} }_{ Geman-McClure }
$$
$$
\underbrace{ \rho(u) = \begin{cases} \frac{1}{2}u^2 & \text{if } |u| \leq \delta \\
\delta|u| - \frac{1}{2}\delta^2 & \text{if } |u| > \delta \end{cases} }_{ \text{Huber (Quad when close, Lin when far)} }
$$
$\delta$ is a specifiable parameter.

These cost functions don't explode as much as the squared loss, and thus are more robust towards outliers. **The downside is that we end up with slower convergence** (but keep in mind that's only if our data has very few outliers)

![[Pasted image 20251115124435.png]]
![[Pasted image 20251115125204.png]]

>[!caution] In the case of nice data, you don't need Robust Loss Functions like these.