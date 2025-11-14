A type of [[Kalman Filter]] that gets rid of the idea of linearizing altogether and instead use [[Sigmapoint (Unscented) Transformation]].

# Setup
Remember, we are trying to solve [[NLNG Problem Statement]]

It can be characterized as:
$$
\text{motion model:}\;\; \mathbf{x}_{k}=\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{v}_{k}, \mathbf{w}_{k}), \;\;k=1\dots K
$$
$$
\text{observation model:}\;\; \mathbf{y}_{k}=\mathbf{g}(\mathbf{x}_{k},\mathbf{n}_{k}), \;\; k=0\dots K
$$
Where $\mathbf{f}(\cdot)$ is a **non-linear motion model** and $\mathbf{g}(\cdot)$ is a **non-linear observation model**.

Where $k$ is a index in discrete time.
- $\mathbf{x}_{k} \in \mathbb{R}^{N}$ is the state of the system 
- $\mathbf{x}_{0} \in \mathbb{R}^{N} \sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{\hat{P}}_{0})$ is the initial state of the system 
- $\mathbf{v}_{k} \in \mathbb{R}^{N}$ input to the system. might have a mapping to $\mathbf{v}_{k}=\mathbf{B}\mathbf{u}_{k}$
- $\mathbf{w}_{k} \in \mathbb{R}^{N}\sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{Q}_{k})$ process noise
- $\mathbf{y}_{k} \in \mathbb{R}^{N}$ measurement
- $\mathbf{n}_{k} \in \mathbb{R}^{N}\sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{R}_{k})$ measurement noise
# Predict 
We stack the previously calculated posterior and our motion noise uncertainty on top of each other.
$$
\boldsymbol{\mu}_{z}=\begin{bmatrix}
\hat{\mathbf{x}}_{k-1} \\
\mathbf{{0}}
\end{bmatrix} \;\; \boldsymbol{\Sigma}_{zz}=\begin{bmatrix}
\hat{\mathbf{P}}_{k-1} & 0 \\
0 & \mathbf{Q}_{k}
\end{bmatrix}
$$
Let $L=dim(\boldsymbol{\mu}_{z})$

We then retrieve the *sigmapoints*
$$
\mathbf{L}\mathbf{L}^{T}=\boldsymbol{\Sigma}_{zz},\;\text{Cholesky decomposition}
$$
$$
\mathbf{z}_{0}=\boldsymbol{\mu}_{z}
$$
$$
\mathbf{z}_{i}=\boldsymbol{\mu}_{z}+\sqrt{ L+\kappa }\text{col}_{i}\mathbf{L}\;\;\; i=0\dots L
$$
$$
\mathbf{z}_{i}=\boldsymbol{\mu}_{z}-\sqrt{ L+\kappa }\text{col}_{i}\mathbf{L}
$$
$$
\alpha_{i}=\begin{cases}
\frac{\kappa}{L+\kappa}, & i=0 \\
\frac{1}{2} \frac{1}{L+\kappa} & otherwise
\end{cases}
$$
Unstack each *sigmapoint* back to state and motion noise
$$
\mathbf{z}_{i}=\begin{bmatrix}
\hat{\mathbf{x}}_{k-1,i} \\
\mathbf{w}_{k,i}
\end{bmatrix}
$$
And then **pass each point through the nonlinear motion model**
$$
\check{\mathbf{x}}_{k,i}=\mathbf{f}(\hat{\mathbf{x}}_{k-1,i},\mathbf{v}_{k},\mathbf{w}_{k,i}) \;\;\; i=0\dots2L
$$
Recombine the transformation to the predicted prior
$$
\check{\mathbf{x}}_{k}=\sum ^{2L}_{i=0}\alpha_{i}\check{\mathbf{x}}_{k,i}
$$
$$
\check{\mathbf{P}}_{k}=\sum ^{2L}_{i=0}\alpha_{i}(\check{\mathbf{x}}_{k,i}-\check{\mathbf{x}}_{k})(\check{\mathbf{x}}_{k,i}-\check{\mathbf{x}}_{k})^{T}
$$
**We now have our predicted priors!**

# Update
Recall from [[Bayes Filter]]
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}}) =\mathcal{N}(\underbrace{ \mu_{x,k}+\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy,k}^{-1}(\mathbf{y}_{k}-\boldsymbol{\mu}_{y,k}) }_{ \hat{\mathbf{x}}_{k} }, \underbrace{ \boldsymbol{\Sigma}_{xx,k}-\boldsymbol{\Sigma}_{xy,k}\boldsymbol{\Sigma}_{yy,k}^{-1}\boldsymbol{\Sigma}_{yx,k} }_{ \hat{\mathbf{P}}_{k} })
$$
As a result, we can write the generalized Gaussian correction-step equations as
$$
\mathbf{K}_{k}=\boldsymbol{\Sigma}_{xy,k}\boldsymbol{\Sigma}_{yy,k}^{-1}
$$
$$
\hat{\mathbf{P}}_{k}=\check{\mathbf{P}}_{k}-\mathbf{K}_{k}\boldsymbol{\Sigma}_{xy,k}^{T}
$$
$$
\hat{\mathbf{x}}_{k}=\check{\mathbf{x}}_{k}+\mathbf{K}_{k}(\mathbf{y}_{k}-\boldsymbol{\mu}_{y,k})
$$

We can use [[Sigmapoint (Unscented) Transformation]] to get values of $\boldsymbol{\mu}_{y,k},\;\boldsymbol{\Sigma}_{yy,k},\;\text{and}\;\boldsymbol{\Sigma}_{xy,k}$.

**First, we use [[Sigmapoint (Unscented) Transformation]] to handle the non-linearity in the observation model.**

We stack the previously calculated posterior and our motion noise uncertainty on top of each other.
$$
\boldsymbol{\mu}_{z}=\begin{bmatrix}
\check{\mathbf{x}}_{k} \\
\mathbf{{0}}
\end{bmatrix} \;\; \boldsymbol{\Sigma}_{zz}=\begin{bmatrix}
\check{\mathbf{P}}_{k} & 0 \\
0 & \mathbf{R}_{k}
\end{bmatrix}
$$
Let $L=dim(\boldsymbol{\mu}_{z})$

We then retrieve the *sigmapoints*
$$
\mathbf{L}\mathbf{L}^{T}=\boldsymbol{\Sigma}_{zz},\;\text{Cholesky decomposition}
$$
$$
\mathbf{z}_{0}=\boldsymbol{\mu}_{z}
$$
$$
\mathbf{z}_{i}=\boldsymbol{\mu}_{z}+\sqrt{ L+\kappa }\text{col}_{i}\mathbf{L}\;\;\; i=0\dots L
$$
$$
\mathbf{z}_{i}=\boldsymbol{\mu}_{z}-\sqrt{ L+\kappa }\text{col}_{i}\mathbf{L}
$$
$$
\alpha_{i}=\begin{cases}
\frac{\kappa}{L+\kappa}, & i=0 \\
\frac{1}{2} \frac{1}{L+\kappa} & otherwise
\end{cases}
$$
Unstack each *sigmapoint* back to state and motion noise
$$
\mathbf{z}_{i}=\begin{bmatrix}
\check{\mathbf{x}}_{k-1,i} \\
\mathbf{n}_{k,i}
\end{bmatrix}
$$
And then **pass each point through the nonlinear motion model**
$$
\check{\mathbf{y}}_{k,i}=\mathbf{g}(\check{\mathbf{x}}_{k-1,i},\mathbf{n}_{k,i}) \;\;\; i=0\dots2L
$$
Recombine the transformation to the predicted prior
$$
\mathbf{\mu}_{y,k}=\sum ^{2L}_{i=0}\alpha_{i}\check{\mathbf{y}}_{k,i}
$$
$$
\mathbf{\Sigma}_{yy,k}=\sum ^{2L}_{i=0}\alpha_{i}(\check{\mathbf{y}}_{k,i}-\mathbf{\boldsymbol{\mu}}_{y,k})(\check{\mathbf{y}}_{k,i}-\mathbf{\boldsymbol{\mu}}_{y,k})^{T}
$$
$$
\mathbf{\Sigma}_{xy,k}=\sum ^{2L}_{i=0}\alpha_{i}(\check{\mathbf{x}}_{k,i}-\mathbf{\boldsymbol{x}}_{k})(\check{\mathbf{y}}_{k,i}-\mathbf{\boldsymbol{\mu}}_{y,k})^{T}
$$
Plug into
$$
\mathbf{K}_{k}=\boldsymbol{\Sigma}_{xy,k}\boldsymbol{\Sigma}_{yy,k}^{-1}
$$
$$
\hat{\mathbf{P}}_{k}=\check{\mathbf{P}}_{k}-\mathbf{K}_{k}\boldsymbol{\Sigma}_{xy,k}^{T}
$$
$$
\hat{\mathbf{x}}_{k}=\check{\mathbf{x}}_{k}+\mathbf{K}_{k}(\mathbf{y}_{k}-\boldsymbol{\mu}_{y,k})
$$

and we have **our updated posterior!**