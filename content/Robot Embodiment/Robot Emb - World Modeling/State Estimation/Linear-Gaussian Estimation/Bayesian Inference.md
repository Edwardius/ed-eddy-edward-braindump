Opposed to [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Maximum A Posteriori]] where we were calculating just the entire state as an optimization problem, Bayesian Inference aims to compute the full posterior (that is, it aims to calculate **the full distribution of probable state trajectories**). 

>[!error] We're not just calculating a single best guess, we are computing the distribution of possible guesses!

This is important because we get an uncertainty value of our estimated state.

# Priors 
From the [[LG Problem Statement]], we see that.
$$
\text{model:} \;\;\mathbf{x}_{k}=\mathbf{A}_{k-1}\mathbf{x}_{k-1}+\mathbf{v}_{k}+\mathbf{w}_{k}
$$
This can be lifted into *lifted matrix form* ([[Matrix and Component Forms]] but we are lifting up component form of matrix expressions to one bigger matrix expression)
$$
\mathbf{x}=\mathbf{A}(\mathbf{v}+\mathbf{w})
$$
$$
\mathbf{A} = \begin{bmatrix} \mathbf{1} & & & & \\ \mathbf{A}_0 & \mathbf{1} & & & \\ \mathbf{A}_1\mathbf{A}_0 & \mathbf{A}_1 & \mathbf{1} & & \\ \vdots & \vdots & \vdots & \ddots & \ddots \\ \mathbf{A}_{K-2}\cdots\mathbf{A}_0 & \mathbf{A}_{K-2}\cdots\mathbf{A}_1 & \mathbf{A}_{K-2}\cdots\mathbf{A}_2 & \cdots & \mathbf{1} \\ \mathbf{A}_{K-1}\cdots\mathbf{A}_0 & \mathbf{A}_{K-1}\cdots\mathbf{A}_1 & \mathbf{A}_{K-1}\cdots\mathbf{A}_2 & \cdots & \mathbf{A}_{K-1} & \mathbf{1} \end{bmatrix}
$$
$\mathbf{A}$ is a **lifted transition matrix** which as shown is **lower-triangular**
The lifted mean, $\check{\mathbf{x}}$, and covariance, $\check{\mathbf{P}}$ , is
$$
\check{\mathbf{x}}=E[\mathbf{x}]=\mathbf{A}\mathbf{v}\;\;\;\; \check{\mathbf{P}}=E[(\mathbf{x}-E[\mathbf{x}])(\mathbf{x}-E[\mathbf{x}])^{T}]=\mathbf{A}\mathbf{Q}\mathbf{A}^{T}
$$
where $\mathbf{Q}=diag(\check{\mathbf{P}},\mathbf{Q}_{1},\dots,\mathbf{Q}_{K})$

We can see that our prior can be neatly represented as:
$$
p(\mathbf{x}|\mathbf{v})=\mathcal{N}(\check{\mathbf{x}},\check{\mathbf{P}})=\mathcal{N}(\mathbf{A}\mathbf{v},\mathbf{A}\mathbf{Q}\mathbf{A}^{T})
$$
This gives us **our priors**

# Posterior
From the [[LG Problem Statement]], we see that
$$
\text{observation model:} \;\;\mathbf{y}_{k}=\mathbf{C}_{k}\mathbf{x}_{k}+\mathbf{n}_{k}
$$
This can be lifted into the form
$$
\mathbf{y}=\mathbf{C}\mathbf{x}+\mathbf{n}
$$
$$
\mathbf{C}=diag(\mathbf{C}_{0},\mathbf{C}_{1},\dots,\mathbf{C}_{K})
$$
We write a [[Joint Gaussian PDFs]]  like so
$$
p(\mathbf{x},\mathbf{y}|\mathbf{v})=\mathcal{N}\left(\begin{bmatrix}
\check{\mathbf{x}} \\
\mathbf{C}\check{\mathbf{x}}
\end{bmatrix}, \begin{bmatrix}
\check{\mathbf{P}} & \check{\mathbf{P}}\mathbf{C}^{T} \\
\mathbf{C}\check{\mathbf{P}} & \mathbf{C}\check{\mathbf{P}}\mathbf{C}^{T}+\mathbf{R}
\end{bmatrix}\right)
$$
$$
\mathbf{R}=E[\mathbf{n}\mathbf{n}^{T}]=diag(\mathbf{R}_0,\mathbf{R}_{1},\dots,\mathbf{R}_{K})
$$
We can factor the joint gaussian to get
$$
p(\mathbf{x},\mathbf{y}|\mathbf{v})=p(\mathbf{x}|\mathbf{v},\mathbf{y})p(\mathbf{y}|\mathbf{v})
$$
We only care about the first factor because that **is the Bayesian Posterior**. We know how to get the normal distribution parameters from the [[Joint Gaussian PDFs]], and using that we get

$$
p(\mathbf{x}|\mathbf{y}) = \mathcal{N}\left(\hat{\mathbf{x}} + \check{\mathbf{P}}\mathbf{C}^T(\mathbf{C}\check{\mathbf{P}}\mathbf{C}^T + \mathbf{R})^{-1}(\mathbf{y} - \mathbf{C}\hat{\mathbf{x}}), \check{\mathbf{P}} - \check{\mathbf{P}}\mathbf{C}^T(\mathbf{C}\check{\mathbf{P}}\mathbf{C}^T + \mathbf{R})^{-1}\mathbf{C}\check{\mathbf{P}}\right)
$$
Using the SMW identity...
$$
p(\mathbf{x}|\mathbf{y}) = \mathcal{N}\left(\underbrace{\left(\check{\mathbf{P}}^{-1} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C}\right)^{-1}\left(\check{\mathbf{P}}^{-1}\check{\mathbf{x}} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{y}\right)}_{\hat{\mathbf{x}}, \text{ mean}}, \underbrace{\left(\check{\mathbf{P}}^{-1} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C}\right)^{-1}}_{\hat{\check{\mathbf{P}}}, \text{ covariance}}\right)
$$
From this we can get an expression for $\hat{\mathbf{x}}$
$$
\hat{\mathbf{x}}=\left(\mathbf{P}^{-1} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C}\right)^{-1}\left(\mathbf{P}^{-1}\check{\mathbf{x}} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{y}\right)
$$
$$
\underbrace{ \left(\mathbf{P}^{-1} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C}\right) }_{ \hat{\mathbf{P}}^{-1} \text{ as shown after SMW} }\hat{\mathbf{x}}=\mathbf{P}^{-1}\check{\mathbf{x}} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{y}
$$
Substituting what we had before 
$$
\check{\mathbf{x}}=\mathbf{A}\mathbf{v}\;\;\;\; \check{\mathbf{P}}=\mathbf{A}\mathbf{Q}\mathbf{A}^{T}
$$
$$
\underbrace{\left(\mathbf{A}^{-T}\mathbf{Q}^{-1}\mathbf{A}^{-1} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C}\right)}_{\mathbf{P}^{-1}} \hat{\mathbf{x}} = \mathbf{A}^{-T}\mathbf{Q}^{-1}\mathbf{v} + \mathbf{C}^T\mathbf{R}^{-1}\mathbf{y}
$$
Computing $\mathbf{A}^{-1}$ turns out to be pretty elegant
$$
\mathbf{A}^{-1} = \begin{bmatrix} \mathbf{1} & & & & \\ -\mathbf{A}_0 & \mathbf{1} & & & \\ & -\mathbf{A}_1 & \mathbf{1} & & \\ & & -\mathbf{A}_2 & \ddots & \\ & & & \ddots & \mathbf{1} \\ & & & & -\mathbf{A}_{K-1} & \mathbf{1} \end{bmatrix}
$$

We can also restack our matricies in a way to make a nicer equation.
$$
\mathbf{z}=\begin{bmatrix}
\mathbf{v} \\
\mathbf{y}
\end{bmatrix} \;\;
\mathbf{H}=\begin{bmatrix}
\mathbf{A}^{-1} \\
\mathbf{C}
\end{bmatrix} \;\;
\mathbf{W}=\begin{bmatrix}
\mathbf{Q} &  \\
 & \mathbf{R}
\end{bmatrix}
$$
Which simplifies the system of equations to.
$$
(\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{H})\hat{\mathbf{x}}=\mathbf{H}^{T}\mathbf{W}^{-1}\mathbf{z}
$$
which is exactly what we saw in [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Maximum A Posteriori]]! This is because we are functioning on Gaussian, whose mean and mode are the same.

>[!error] Important to note here that Bayesian Inference lets us retreive the **mean** easier