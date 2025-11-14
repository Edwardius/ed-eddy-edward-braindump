Following [[Multivariate Gaussian]], we can also design a joint PDF of two multivariate gaussians.
$$
p(\mathbf{x},\mathbf{y})=\mathcal{N}\left(\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}, \begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}\right)
$$
$$
\boldsymbol{\Sigma}_{xy}=\boldsymbol{\Sigma}_{yx}^{T}
$$
Which if expanded looks like
$$
\frac{1}{(2\pi)^{d/2}\begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}^{1/2}}\exp\left( -\frac{1}{2}\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right)^{T}\begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}^{-1}\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right) \right)
$$

We can always represent a [[Basic Probability Nomenclature|joint probability]] as the product of two factors
$$
p(\mathbf{x},\mathbf{y})=p(\mathbf{x}|\mathbf{y})p(\mathbf{y})
$$
So you can definitely split a Joint Gaussian into something similar. To do so, we need to use something called the [[Schur Complement]]
$$
\begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}=\begin{bmatrix}
\mathbf{1}  & \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1} \\
\mathbf{0}  & \mathbf{1}
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx} & \mathbf{0} \\
\mathbf{0} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}
\begin{bmatrix}
\mathbf{1}  & \mathbf{0} \\
\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx}  & \mathbf{1}
\end{bmatrix}
$$
Inverting this we get
$$
\begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}^{-1}=\begin{bmatrix}
\mathbf{1}  & \mathbf{0}\\
-\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx}   & \mathbf{1}
\end{bmatrix}
\begin{bmatrix}
(\boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx})^{-1} & \mathbf{0} \\
\mathbf{0} & \boldsymbol{\Sigma}_{yy}^{-1}
\end{bmatrix}
\begin{bmatrix}
\mathbf{1}  & -\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1} \\
\mathbf{0}  & \mathbf{1}
\end{bmatrix}
$$
If we use this to analyze the quadratic part of the Gaussian PDF...
$$
\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right)^{T}\begin{bmatrix}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{bmatrix}^{-1}\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right)
$$
$$
=\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right)^{T}
\begin{bmatrix}
\mathbf{1}  & \mathbf{0}\\
-\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx}   & \mathbf{1}
\end{bmatrix}
\begin{bmatrix}
(\boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx})^{-1} & \mathbf{0} \\
\mathbf{0} & \boldsymbol{\Sigma}_{yy}^{-1}
\end{bmatrix}
\begin{bmatrix}
\mathbf{1}  & -\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1} \\
\mathbf{0}  & \mathbf{1}
\end{bmatrix}\left(\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{\mu}_{x} \\
\boldsymbol{\mu}_{y}
\end{bmatrix}\right)
$$

$$= \underbrace{ (\mathbf{x} - \boldsymbol{\mu}_{\mathbf{x}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y}} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} (\mathbf{y} - \boldsymbol{\mu}_{\mathbf{y}}))^{T} (\boldsymbol{\Sigma}_{\mathbf{x} \mathbf{x}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y}} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{x}})^{-1}(\mathbf{x} - \boldsymbol{\mu}_{\mathbf{x}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y}} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} (\mathbf{y} - \boldsymbol{\mu}_{\mathbf{y}})) }_{ p(\mathbf{x}|\mathbf{y}) } + \underbrace{ (\mathbf{y} - \boldsymbol{\mu}_{\mathbf{y}})^{T} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} (\mathbf{y} - \boldsymbol{\mu}_{\mathbf{y}}) }_{ p(\mathbf{y}) }$$
^^ Because we are looking at the quadratic part of the Joint Gaussian and addition means multiplication!!

**This gets us the following breakdown of the Joint Gaussian Distribution**

$$
p(\mathbf{x},\mathbf{y})=p(\mathbf{x}|\mathbf{y})p(\mathbf{y})
$$
$$
p(\mathbf{x}|\mathbf{y})=\mathcal{N}(\boldsymbol{\mu}_{\mathbf{x}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y}} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} (\mathbf{y} - \boldsymbol{\mu}_{\mathbf{y}}), \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{x}} - \boldsymbol{\Sigma}_{\mathbf{x} \mathbf{y}} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} \boldsymbol{\Sigma}_{\mathbf{y} \mathbf{x}})
$$
$$
p(\mathbf{y})=\mathcal{N}(\boldsymbol{\mu}_{y},\boldsymbol{\Sigma}_{yy})
$$
