Constructs a local optimization through [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] for a small window of estimates and states.

$$
\underbrace{\begin{bmatrix} \tilde{\mathbf{A}}_{kk} & \mathbf{A}_{k+1,k}^T \\ \mathbf{A}_{k+1,k} & \mathbf{A}_{k+1,k+1} & \mathbf{A}_{k+2,k+1}^T \\ & \mathbf{A}_{k+2,k+1} & \mathbf{A}_{k+2,k+2} & \mathbf{A}_{k+3,k+2}^T \\ & & \mathbf{A}_{k+3,k+2} & \mathbf{A}_{k+3,k+3} \end{bmatrix}}_{\mathbf{H}^T\mathbf{W}^{-1}\mathbf{H}} \underbrace{\begin{bmatrix} \delta\mathbf{x}_k^* \\ \delta\mathbf{x}_{k+1}^* \\ \delta\mathbf{x}_{k+2}^* \\ \delta\mathbf{x}_{k+3}^* \end{bmatrix}}_{\delta\mathbf{x}^*} = \underbrace{\begin{bmatrix} \tilde{\mathbf{b}}_k \\ \mathbf{b}_{k+1} \\ \mathbf{b}_{k+2} \\ \mathbf{b}_{k+3} \end{bmatrix}}_{\mathbf{H}^T\mathbf{W}^{-1}\mathbf{e}}
$$


$$
\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{x}_0 \\ \delta\mathbf{x}_1 \\ \delta\mathbf{x}_2 \\ \vdots \\ \delta\mathbf{x}_K \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} \mathbf{1} & & & & \\ -\mathbf{F}_0 & \mathbf{1} & & & \\ & -\mathbf{F}_1 & \ddots & & \\ & & \ddots & \mathbf{1} & \\ & & & -\mathbf{F}_{K-1} & \mathbf{1} \\ \hline -\mathbf{G}_0 & & & & \\ & \mathbf{G}_1 & & & \\ & & \mathbf{G}_2 & & \\ & & & \ddots & \\ & & & & \mathbf{G}_K \end{bmatrix}
$$
$$
\mathbf{e}(\mathbf{x}_{\text{op}}) = \begin{bmatrix} \mathbf{e}_{v,0}(\mathbf{x}_{\text{op}}) \\ \mathbf{e}_{v,1}(\mathbf{x}_{\text{op}}) \\ \vdots \\ \mathbf{e}_{v,K}(\mathbf{x}_{\text{op}}) \\ \hline \mathbf{e}_{y,0}(\mathbf{x}_{\text{op}}) \\ \mathbf{e}_{y,1}(\mathbf{x}_{\text{op}}) \\ \vdots \\ \mathbf{e}_{y,K}(\mathbf{x}_{\text{op}}) \end{bmatrix}
$$
$$
\mathbf{W} = \text{diag}\left(\mathbf{P}_0, \mathbf{Q}_1, \ldots, \mathbf{Q}_K, \mathbf{R}_0, \mathbf{R}_1, \ldots, \mathbf{R}_K\right)
$$
You're just constructing a local optimization problem to iterate with [[Gauss-Newton Method]] on. 

**The only special part is $\bar{\mathbf{A}}_{kk}$ and $\bar{\mathbf{b}}_{k}$**. which are specifically defined as 

$$
\bar{\mathbf{A}}_{kk} = \mathbf{P}_k^{-1} + \mathbf{F}_k^T\mathbf{Q}_{k+1}'^{-1}\mathbf{F}_k + \mathbf{G}_k^T\mathbf{R}_k'^{-1}\mathbf{G}_k
$$
$$
\bar{\mathbf{b}}_k = \mathbf{c}_k + \mathbf{P}_k^{-1}\mathbf{e}_{v,k} - \mathbf{F}_k^T\mathbf{Q}_{k+1}'^{-1}\mathbf{e}_{v,k+1} + \mathbf{G}_k^T\mathbf{R}_k'^{-1}\mathbf{e}_{y,k}
$$
$$
\mathbf{P}_k^{-1} = \mathbf{Q}_k'^{-1} - \mathbf{Q}_k'^{-1}\mathbf{F}_{k-1}\mathbf{A}_{k-1,k-1}^{-1}\mathbf{F}_{k-1}^T\mathbf{Q}_{k'^{-1}}\quad\text{previous timestep k-1}
$$
$$
\mathbf{c}_k = \mathbf{Q}_k'^{-1}\mathbf{F}_{k-1}\mathbf{A}_{k-1,k-1}^{-1}\left(\mathbf{c}_{k-1} + \mathbf{P}_{k-1}^{-1}\mathbf{e}_{v,k-1} + \mathbf{G}_{k-1}^T\mathbf{R}_{k-1}'^{-1}\mathbf{e}_{y,k-1}\right)\quad\text{previous timestep k-1}
$$
Where
$$
\mathbf{c}_{0}=\mathbf{0}\quad \check{\mathbf{P}}_{0}^{-1}\text{ the provided initial information matrix}
$$
