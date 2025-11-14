This follows [[Robot Embodiment/World Modeling/State Estimation/Linear-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] in Linear-Gaussian Estimation, except for the nonlinear case.

We previously set the objective function to be the squared [[Mahalanobis Distance]] . Here we define the errors between the prior and measurements differently...
$$
\mathbf{e}_{v,k}(\mathbf{x})=\begin{cases}
\check{\mathbf{x}}_{0}-\mathbf{x}_{0} & k=0 \\
\mathbf{f}(\mathbf{x}_{k-1},\mathbf{v}_{k},\mathbf{0}) & k=1\dots K
\end{cases}
$$$$
\mathbf{e}_{y,k}(\mathbf{x})=\mathbf{y}_{k}-\mathbf{g}(\mathbf{x}_{k},\mathbf{0})\;\; k=0\dots K
$$
We define their contributions to the objective function as
$$
J_{v,k}(\mathbf{x})=\frac{1}{2}\mathbf{e}_{v,k}(\mathbf{x})^{T}\mathbf{W}_{v,k}^{-1}\mathbf{e}_{v,k}(\mathbf{x})
$$
$$
J_{y,k}(\mathbf{x})=\frac{1}{2}\mathbf{e}_{y,k}(\mathbf{x})^{T}\mathbf{W}_{y,k}^{-1}\mathbf{e}_{y,k}(\mathbf{x})
$$
>[!error] $\mathbf{W}_{v,k}$ and $\mathbf{W}_{y,k}$ can be thought of as positive-definite symmetric matrix weights **that are often set to the process noise and measurement noise covariances of the system**

And the overall objective function is thus
$$
J(\mathbf{x})=\sum ^{K}_{k=0}(J_{v,k}(\mathbf{x})+J_{y,k}(\mathbf{x}))
$$
We can rewrite this to be cleaner
$$
\mathbf{e}(\mathbf{x}) = \begin{bmatrix} \mathbf{e}_v(\mathbf{x}) \\ \mathbf{e}_y(\mathbf{x}) \end{bmatrix}, \quad \mathbf{e}_v(\mathbf{x}) = \begin{bmatrix} \mathbf{e}_{v,0}(\mathbf{x}) \\ \vdots \\ \mathbf{e}_{v,K}(\mathbf{x}) \end{bmatrix}, \quad \mathbf{e}_y(\mathbf{x}) = \begin{bmatrix} \mathbf{e}_{y,0}(\mathbf{x}) \\ \vdots \\ \mathbf{e}_{y,K}(\mathbf{x}) \end{bmatrix}
$$

$$
\mathbf{W} = \text{diag}(\mathbf{W}_v, \mathbf{W}_y), \quad \mathbf{W}_v = \text{diag}(\mathbf{W}_{v,0}, \ldots, \mathbf{W}_{v,K})
$$

$$
\mathbf{W}_y = \text{diag}(\mathbf{W}_{y,0}, \ldots, \mathbf{W}_{y,K})
$$
so that the objective function can be written as
$$
J(\mathbf{x}) = \frac{1}{2}\mathbf{e}(\mathbf{x})^T \mathbf{W}^{-1} \mathbf{e}(\mathbf{x})
$$
We can further define the modified error term,
$$
\mathbf{u}(\mathbf{x}) = \mathbf{L}\mathbf{e}(\mathbf{x})
$$
where $\mathbf{L}^T\mathbf{L} = \mathbf{W}^{-1}$ (i.e., from a Cholesky decomposition since $\mathbf{W}$ is symmetric positive-definite). Using these definitions, we can write the objective function simply as
$$
J(\mathbf{x}) = \frac{1}{2}\mathbf{u}(\mathbf{x})^T \mathbf{u}(\mathbf{x})
$$
And we end up with the final goal of
$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmin}}\;J(\mathbf{x})
$$
There are many ways to solve this optimization problem, including [[Newton's Method]] and [[Gauss-Newton Method]]

# [[Gauss-Newton Method]] in Terms of Errors
From the example in [[Gauss-Newton Method]], we get
$$
\Rightarrow \quad \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) \delta\mathbf{x}^* = -\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \mathbf{u}(\mathbf{x}_{\text{op}})
$$
We have that the error is related to $\mathbf{u}(x)$:
$$
\mathbf{u}(\mathbf{x}) = \mathbf{L}\mathbf{e}(\mathbf{x})
$$
So ideally we should express finding our step in terms of the error.
$$
(\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{H})\delta \mathbf{x}^{*}=\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{e}(\mathbf{x}_{op}) \quad \mathbf{H}=-\frac{ \partial \mathbf{e}(\mathbf{x}) }{ \partial \mathbf{x} } \bigg|_{\mathbf{x}_{op}}
$$
Here, we **gotta linearize $\mathbf{e}(\mathbf{x})$** instead
$$
\mathbf{e}_{v,k}(\mathbf{x}_{\text{op}} + \delta\mathbf{x}) \approx \begin{cases} \mathbf{e}_{v,0}(\mathbf{x}_{\text{op}}) - \delta\hat{\mathbf{x}}_0, & k = 0 \\ \mathbf{e}_{v,k}(\mathbf{x}_{\text{op}}) + \mathbf{F}_{k-1}\delta\mathbf{x}_{k-1} - \delta\mathbf{x}_k, & k = 1 \ldots K \end{cases}
$$
$$
\mathbf{e}_{y,k}(\mathbf{x}_{\text{op}} + \delta\mathbf{x}) \approx \mathbf{e}_{y,k}(\mathbf{x}_{\text{op}}) - \mathbf{G}_k\delta\mathbf{x}_k, \quad k = 0 \ldots K
$$
where
$$
\mathbf{e}_{v,k}(\mathbf{x}_{\text{op}}) \approx \begin{cases} \hat{\mathbf{x}}_0 - \mathbf{x}_{\text{op},0}, & k = 0 \\ \mathbf{f}(\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, 0) - \mathbf{x}_{\text{op},k}, & k = 1 \ldots K \end{cases}
$$
$$
\mathbf{e}_{y,k}(\mathbf{x}_{\text{op}}) \approx \mathbf{y}_k - \mathbf{g}(\mathbf{x}_{\text{op},k}, 0), \quad k = 0 \ldots K
$$
$$
\mathbf{F}_{k-1} = \frac{\partial \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{v}_k, \mathbf{w}_k)}{\partial \mathbf{x}_{k-1}}\bigg|_{\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, 0}, \quad \mathbf{G}_k = \frac{\partial \mathbf{g}(\mathbf{x}_k, \mathbf{n}_k)}{\partial \mathbf{x}_k}\bigg|_{\mathbf{x}_{\text{op},k}, 0}
$$
we let 
$$
\mathbf{W}_{v,k}=\mathbf{Q}_{k}'\quad\mathbf{W}_{y,k}=\mathbf{R}_{k}'
$$
With all this in mind, we can construct our step equation for [[Gauss-Newton Method]] by cleverly stacking things
$$
\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{x}_0 \\ \delta\mathbf{x}_1 \\ \delta\mathbf{x}_2 \\ \vdots \\ \delta\mathbf{x}_K \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} \mathbf{1} & & & & \\ -\mathbf{F}_0 & \mathbf{1} & & & \\ & -\mathbf{F}_1 & \ddots & & \\ & & \ddots & \mathbf{1} & \\ & & & -\mathbf{F}_{K-1} & \mathbf{1} \\ \hline -\mathbf{G}_0 & & & & \\ & \mathbf{G}_1 & & & \\ & & \mathbf{G}_2 & & \\ & & & \ddots & \\ & & & & \mathbf{G}_K \end{bmatrix}
$$
$$
\mathbf{e}(\mathbf{x}_{\text{op}}) = \begin{bmatrix} \mathbf{e}_{v,0}(\mathbf{x}_{\text{op}}) \\ \mathbf{e}_{v,1}(\mathbf{x}_{\text{op}}) \\ \vdots \\ \mathbf{e}_{v,K}(\mathbf{x}_{\text{op}}) \\ \hline \mathbf{e}_{y,0}(\mathbf{x}_{\text{op}}) \\ \mathbf{e}_{y,1}(\mathbf{x}_{\text{op}}) \\ \vdots \\ \mathbf{e}_{y,K}(\mathbf{x}_{\text{op}}) \end{bmatrix}
$$
$$
\mathbf{W} = \text{diag}\left(\mathbf{P}_0, \mathbf{Q}_1, \ldots, \mathbf{Q}_K, \mathbf{R}_0, \mathbf{R}_1, \ldots, \mathbf{R}_K\right)
$$
Which lets us yield our [[Gauss-Newton Method|Gauss-Newton Update]]
$$
(\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{H})\delta \mathbf{x}^{*}=\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{e}(\mathbf{x}_{op})
$$
# Laplace Approximation
We sometimes want a covariance matrix of our approximation once its done. To do that we approximate it as the inverse of the approximated hessian at the point where we stopped convergence.

$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmin}}\;J(\mathbf{x})
$$
$$
\check{\mathbf{P}}=\left(\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)\right)^{-1}=(\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{H})^{-1}
$$
