For a rotation matrix $\mathbf{C}$ defined in [[Rotation Representations]] by the Euler rotational theorem
$$
\mathbf{C}_{21}=\cos \phi \mathbf{1}+(1-\cos \phi)\mathbf{a}\mathbf{a}^{T}-\sin \phi \mathbf{a}^{\times}
$$
The partial derivative with respect to the amount of rotation is given by
$$
\frac{ \partial \mathbf{C} }{ \partial \phi }=-\mathbf{a}^{\times}\mathbf{C} 
$$
# Euler Angles
Say we are doing an arbitrary Euler sequence (from [[Rotation Representations]], like how we said 1-2-3). We can get a partial derivative as
$$
\frac{ \partial \mathbf{C}(\boldsymbol{\theta})\mathbf{v} }{ \partial \boldsymbol{\theta} } =(\mathbf{C}(\boldsymbol{\theta})\mathbf{v})^{\times}\underbrace{ \begin{bmatrix}
\mathbf{C}_{\gamma}(\theta_{3})\mathbf{C}_{\beta}(\theta_{2})\mathbf{1}_{\alpha} & \mathbf{C}_{\gamma}(\theta_{3})\mathbf{1}_{\beta}  & \mathbf{1}_{\gamma}
\end{bmatrix} }_{ \mathbf{S}(\theta_{2}, \theta_{3}) }
$$
