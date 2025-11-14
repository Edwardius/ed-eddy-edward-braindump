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
We can always represent a [[Basic Probability Nomenclature|joint probability]] as the product of two factors
$$
p(\mathbf{x},\mathbf{y})=p(\mathbf{x}|\mathbf{y})p(\mathbf{y})
$$
So you can definitely split a Joint Gaussian into something similar. To do so, we need to use something all the [[Schur Complement]]