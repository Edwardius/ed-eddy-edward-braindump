[[Bayes Filter]] Provides us a fundamental framework for designing frameworks. However, it is very high-level, and its generalized to all PDF. We don't need something so generalized in Engineering ;) so we can look specifically at a subset of filters that **assume Gaussian [[Basic Probability Nomenclature|PDFs]]** up front.

Recall Bayes Filter is
$$
\underbrace{ p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}}) }_{ \text{posterior belief} }=

\eta \underbrace{ p(\mathbf{y}_{k}|\mathbf{x}_{k}) }_{ \substack{\text{observation} \\ \text{correction} \\ \text{using}\;\mathbf{g}(\cdot)} }

\int \underbrace{ p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) }_{ \substack{\text{motion prediction} \\ \text{using }\mathbf{f}(\cdot)} }

\underbrace{ p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1}) }_{ \text{prior belief} }d\mathbf{x}_{k-1}
$$
# Prediction
In general, we begin by assuming a Gaussian prior at time $k-1$
$$
p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1})=\mathcal{N}(\hat{\mathbf{x}}_{k-1},\hat{\mathbf{P}}_{k-1})
$$
We the assume that passing this though a non-linear motion model $\mathbf{f}(\cdot)$ is gonna give us another Gaussian
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{k},\mathbf{y}_{0:k-1})=\mathcal{N}(\check{\mathbf{x}}_{k},\check{\mathbf{P}}_{k})
$$
This is our **prediction prior**
# Update
Let's assume that our posterior is going to be Gaussian in nature.

From [[Joint Gaussian PDFs]] we can write a Joint Gaussian with the state and the measurement.
$$
p(\mathbf{x}_{k}, \mathbf{y}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{0:k-1})=\mathcal{N}\left(\begin{bmatrix}
\boldsymbol{\mu}_{x,k} \\
\boldsymbol{\mu}_{y,k}
\end{bmatrix}, \begin{bmatrix}
\boldsymbol{\Sigma}_{xx,k} & \boldsymbol{\Sigma}_{xy,k} \\
\boldsymbol{\Sigma}_{yx,k} & \boldsymbol{\Sigma}_{yy,k}
\end{bmatrix}\right)
$$
From what we saw in [[Joint Gaussian PDFs]] we can split up the joint probability into two factors, both being Gaussian with means and covariances comprised of the factorized quadratic part of the joint distribution
$$
p(\mathbf{x}_{k}, \mathbf{y}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{0:k-1})=p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k})p(\mathbf{y}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{0:k-1})
$$
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k})=\mathcal{N}(\underbrace{ \boldsymbol{\mu}_{\mathbf{x,k}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y},k} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y},k}^{-1} (\mathbf{y}_{k} - \boldsymbol{\mu}_{\mathbf{y,k}}) }_{ \hat{\mathbf{x}}_{k} }, \underbrace{ \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{x},k} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y},k} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y},k}^{-1} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{x},k} }_{ \hat{\mathbf{P}}_{k} })
$$
Lining this up with our prediction prior
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{k},\mathbf{y}_{0:k-1})=\mathcal{N}(\check{\mathbf{x}}_{k},\check{\mathbf{P}}_{k})
$$
We see that a **generalized Gaussian correction-step** appears that could bridge our posterior and prediction prior together.
$$
\mathbf{K}_{k}=\boldsymbol{\Sigma}_{xy,k}\boldsymbol{\Sigma}_{yy,k}^{-1}
$$
$$
\hat{\mathbf{P}}_{k}=\check{\mathbf{P}}_{k}-\mathbf{K}_{k}\boldsymbol{\Sigma}_{xy,k}^{T}
$$
$$
\hat{\mathbf{x}}_{k}=\check{\mathbf{x}}_{k}+\mathbf{K}_{k}(\mathbf{y}_{k}-\boldsymbol{\mu}_{y,k})
$$
