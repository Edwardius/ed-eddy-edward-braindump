Its the [[Sigmapoint (Unscented) Kalman Filter]] except **instead of getting your posterior and moving onto the next timestep, take that posterior and turn it into your predicted prior and keep doing this until some threshold**

# In the Update Step ONLY
First iteration
$$
\boldsymbol{\mu}_{z}=\begin{bmatrix}
\check{\mathbf{x}}_{k} \\
\mathbf{{0}}
\end{bmatrix} \;\; \boldsymbol{\Sigma}_{zz}=\begin{bmatrix}
\check{\mathbf{P}}_{k} & 0 \\
0 & \mathbf{R}_{k}
\end{bmatrix}
$$
$$
\mathbf{x}_{op,k}\leftarrow \hat{\mathbf{x}}_{k}
$$
Second iteration
$$
\boldsymbol{\mu}_{z}=\begin{bmatrix}
\mathbf{x}_{op,k} \\
\mathbf{{0}}
\end{bmatrix} \;\; \boldsymbol{\Sigma}_{zz}=\begin{bmatrix}
\check{\mathbf{P}}_{k} & 0 \\
0 & \mathbf{R}_{k}
\end{bmatrix}
$$
$$
\mathbf{x}_{op,k}\leftarrow \hat{\mathbf{x}}_{k}
$$
Third iteration

$$
\boldsymbol{\mu}_{z}=\begin{bmatrix}
\mathbf{x}_{op,k} \\
\mathbf{{0}}
\end{bmatrix} \;\; \boldsymbol{\Sigma}_{zz}=\begin{bmatrix}
\check{\mathbf{P}}_{k} & 0 \\
0 & \mathbf{R}_{k}
\end{bmatrix}
$$
$$
\mathbf{x}_{op,k}\leftarrow \hat{\mathbf{x}}_{k}
$$
...
Stop once $\Delta \mathbf{x}_{op,k}<threshold$
