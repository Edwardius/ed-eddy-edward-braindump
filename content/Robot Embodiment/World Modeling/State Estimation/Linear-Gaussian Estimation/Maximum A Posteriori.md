The goal of MAP is generally:
$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmax}}\;p(\mathbf{x}|\mathbf{v},\mathbf{y})
$$
$$
\mathbf{x}=\mathbf{x}_{0:K},\;\;\mathbf{v}=(\check{\mathbf{x}}_{0},\mathbf{v}_{1:K}), \;\;\mathbf{y}=\mathbf{y}_{0:K}
$$
Which means that we are trying to find the single best estimate for the state of the system given our control input and measurements.

Using [[Bayes' Theorem]]
$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmax}}\;p(\mathbf{x}|\mathbf{v},\mathbf{y})=\underset{\mathbf{x}}{\text{argmax}}\;\frac{p(\mathbf{y}|\mathbf{x},\mathbf{v})p(\mathbf{x}|\mathbf{v})}{p(\mathbf{y}|\mathbf{v})}=\underset{\mathbf{x}}{\text{argmax}}\;p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{v})
$$
>[!error] yes you can single out specific variables and keep the givens, here was a singling out $x$ and $y$ and how they interact with each other 

We can drop the denominator because it doesn't depend on $\mathbf{x}$ (we are trying to argmax here). We can drop $\mathbf{v}$ because $\mathbf{y}$ doesn't depend on it. (see observation model in [[LG Problem Statement]]).

Each set of state and measurement is independent of the other sets, so
$$
p(\mathbf{y}|\mathbf{x})=\prod_{k=0}^{K}p(\mathbf{y}_{k}|\mathbf{x}_{k}) 
$$
Looking at our motion model
$$
\text{model:} \;\;\mathbf{x}_{k}=\mathbf{A}_{k-1}\mathbf{x}_{k-1}+\mathbf{v}_{k}+\mathbf{w}_{k}
$$
we see that $\mathbf{x}_{k}$ depends on its previous state and the input. As a result, we can factor $\mathbf{p}(\mathbf{x}|\mathbf{v})$ as
$$
p(\mathbf{x}|\mathbf{v})=p(\mathbf{x}_{0}|\check{\mathbf{x}}_{0})\prod_{k=1}^{K}p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) 
$$
Which gives us
$$
\underset{\mathbf{x}}{\text{argmax}}\;p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{v})=\underset{\mathbf{x}}{\text{argmax}}\;p(\mathbf{x}_{0}|\check{\mathbf{x}}_{0})\prod_{k=0}^{K}p(\mathbf{y}_{k}|\mathbf{x}_{k})\prod_{k=1}^{K}p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) 
$$
From the [[LG Problem Statement]] its its motion and observation models, we can get that:
$$
p(\mathbf{x}_{0}|\check{\mathbf{x}_{0}})\sim\mathcal{N}(\check{\mathbf{x}}_{0},\check{\mathbf{P}}_{0})
$$
$$
p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k})\sim\mathcal{N}(\mathbf{A}_{k-1}\mathbf{x}_{k-1}+\mathbf{v}_{k},\mathbf{Q}_{k})
$$
$$
p(\mathbf{y}_{k}|\mathbf{x}_{k})\sim\mathcal{N}(\mathbf{C}_{k}\mathbf{x}_{k},\mathbf{R}_{k})
$$
**To make optimization easier, the logarithm of both sides is taken**
$$
\underset{\mathbf{x}}{\text{argmax}}\;\ln(p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{v})) = \underset{\mathbf{x}}{\text{argmax}}\;\ln p(\mathbf{x}_0|\check{\mathbf{x}}_0) + \sum_{k=1}^{K} \ln p(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{v}_k) + \sum_{k=0}^{K} \ln p(\mathbf{y}_k|\mathbf{x}_k)
$$
where

$$
\ln p(\mathbf{x}_0|\check{\mathbf{x}}_0) = -\frac{1}{2}(\mathbf{x}_0 - \check{\mathbf{x}}_0)^T \mathbf{P}_0^{-1}(\mathbf{x}_0 - \check{\mathbf{x}}_0) \underbrace{ - \cancelto{ 0 }{ \frac{1}{2}\ln((2\pi)^N \det \mathbf{P}_0) } }_{ \text{independant of x} }
$$
$$
\ln p(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{v}_k) = -\frac{1}{2}(\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k)^T \mathbf{Q}_k^{-1}(\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k) - \underbrace{ \cancelto{ 0 }{ \frac{1}{2}\ln((2\pi)^N \det \mathbf{Q}_k) } }_{ \text{independant of x} }
$$
$$
\ln p(\mathbf{y}_k|\mathbf{x}_k) = -\frac{1}{2}(\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)^T \mathbf{R}_k^{-1}(\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k) - \underbrace{ \cancelto{ 0 }{ \frac{1}{2}\ln((2\pi)^M \det \mathbf{R}_k) } }_{ \text{independant of x} }
$$
After cancelling the terms that are independent of x, we can construct an objective function.
$$
J(\mathbf{x})=\sum ^{K}_{k=0}(J_{v,k}(\mathbf{x})+J_{y,k}(\mathbf{x}))
$$
$$
J_{v,k}(\mathbf{x})=\begin{cases}
\frac{1}{2}(\mathbf{x}_0 - \check{\mathbf{x}}_0)^T \mathbf{P}_0^{-1}(\mathbf{x}_0 - \check{\mathbf{x}}_0) & k=0 \\
\frac{1}{2}(\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k)^T \mathbf{Q}_k^{-1}(\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k) & k=1\dots K
\end{cases}
$$
$$
J_{y,k}(\mathbf{x})=\frac{1}{2}(\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)^T \mathbf{R}_k^{-1}(\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)
$$
>[!info] This objective function is directly grabbed from the logged gaussians you derived above. The negative has been removed so it becomes a minimization problem

**So we end up with a simplified, equivalent optimization problem**
$$
\hat{\mathbf{x}}=\underset{\mathbf{x}}{\text{argmin}}\;J(\mathbf{x})
$$
We can convert our objective function into a cleaner [[Matrix and Component Forms|Matrix Form]] by cleverly stacking our known data.
$$
\mathbf{z} = \begin{bmatrix} \mathbf{x}_0 \\ \mathbf{v}_1 \\ \vdots \\ \mathbf{v}_K \\ \mathbf{y}_0 \\ \mathbf{y}_1 \\ \vdots \\ \mathbf{y}_K \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} \mathbf{x}_0 \\ \vdots \\ \mathbf{x}_K \end{bmatrix}
$$
We then define the following block-matrix quantities:
$$
\mathbf{H} = \begin{bmatrix} \mathbf{1} & & & \\ -\mathbf{A}_0 & \mathbf{1} & & \\ & \ddots & \ddots & \\ & & -\mathbf{A}_{K-1} & \mathbf{1} \\ \hline \mathbf{C}_0 & & & \\ & \mathbf{C}_1 & & \\ & & \ddots & \\ & & & \mathbf{C}_K \end{bmatrix}
$$
$$
\mathbf{W} = \begin{bmatrix} \mathbf{P}_0 & & & \\ & \mathbf{Q}_1 & & \\ & & \ddots & \\ & & & \mathbf{Q}_K \\ \hline & & & & \mathbf{R}_0 & & & \\ & & & & & \mathbf{R}_1 & & \\ & & & & & & \ddots & \\ & & & & & & & \mathbf{R}_K \end{bmatrix}
$$
This lets us represent our objective function as:
$$
J(\mathbf{x})=\frac{1}{2}(\mathbf{z}-\mathbf{H}\mathbf{x})^{T}\mathbf{W}^{-1}(\mathbf{z}-\mathbf{H}\mathbf{x})
$$
Since $J(\mathbf{x})$ is a paraboloid, there exists a closed form solution if we set its partial derivative with respect to $\mathbf{x}$ to 0.
$$
\frac{ \partial J(\mathbf{x}) }{ \partial \mathbf{x}^{T} } |_{\hat{\mathbf{x}}}=0
$$
$$
-\mathbf{H}^{T}\mathbf{W}^{-1}(\mathbf{z}-\mathbf{H}\hat{\mathbf{x}})-0
$$
$$
(\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{H})\hat{\mathbf{x}}=\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{z}
$$
Which is a [[Normal Equation]]!

>[!error] It is important to note here that Maximum A Posteriori will optimize to  the **mode** of the distribution!