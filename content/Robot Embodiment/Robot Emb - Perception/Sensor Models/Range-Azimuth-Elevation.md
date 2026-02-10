Common model for sensors like LiDAR
![[Pasted image 20251115203353.png]]
$$
\boldsymbol{\rho}=\mathbf{r}_{s}^{ps}=\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
$$
The sensor itself gets readying by RAE, but we get it in 3D points
$$
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}=\begin{bmatrix}
r\cos \alpha \cos \epsilon \\
r\sin \alpha \cos\epsilon \\
r\sin\epsilon
\end{bmatrix}
$$
This is the inverse of the sensor model, so the sensor model is actually
$$
\begin{bmatrix}
r \\
\alpha \\
\epsilon
\end{bmatrix}=\mathbf{s}(\boldsymbol{\rho})=\begin{bmatrix}
\sqrt{ x^{2}+y^{2}+z^{2} } \\
\tan ^{-1}\left( \frac{y}{x} \right) \\
\sin ^{-1}\left( \frac{z}{\sqrt{ x^{2}+y^{2}+z^{2} }} \right)
\end{bmatrix}
$$
For 2D lidars, we set $z=0$