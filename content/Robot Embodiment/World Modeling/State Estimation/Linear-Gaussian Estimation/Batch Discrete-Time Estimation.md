Solves the entire [[LG Problem Statement]] as a linear least squares ([[Normal Equation]]) problem. It uses all the data we have at once.

**Construct Matrices of the Data**
$$
\mathbf{x}=\begin{bmatrix}
\mathbf{x}_{0} \\
\mathbf{x}_{1} \\
\dots \\
\mathbf{x}_{K}
\end{bmatrix}

\;\;

\mathbf{y}=\begin{bmatrix}
\mathbf{y}_{0} \\
\mathbf{y}_{1} \\
\dots \\
\mathbf{y}_{K}
\end{bmatrix}

\;\; \mathbf{v}=\begin{bmatrix}
\mathbf{v}_{0} \\
\mathbf{v}_{1} \\
\dots \\
\mathbf{v}_{K}
\end{bmatrix}
$$
$$
\text{Dynamics Matrix:}\;\mathbf{A}=\begin{bmatrix}
-\mathbf{I} & 0  & 0& \dots & 0 \\
\mathbf{A}_{0} & -\mathbf{I}  & 0& \dots & 0 \\
0 & \mathbf{A}_{1}  & -\mathbf{I}& \dots & 0 \\
\vdots & \vdots &  & \vdots  & \vdots\\
0 & 0 & 0 & \dots & -\mathbf{I}
\end{bmatrix}
$$
$$
\text{Observation Matrix:} \;\mathbf{C}=blockdiag(\mathbf{C}_{1},\mathbf{C}_{2},\dots,\mathbf{C}_{K})=\begin{bmatrix}
\mathbf{C}_{1} & 0 & \dots & 0 \\
0 & \mathbf{C}_{1} & \dots & 0 \\
\vdots & \vdots &  & \vdots \\
0 & 0 & \dots & \mathbf{C}_{K}
\end{bmatrix}
$$
$$
\mathbf{R}=blockdiag(\mathbf{R}_{1},\mathbf{R}_{2},\dots ,\mathbf{R}_{K})
$$
$$
\mathbf{Q}=blockdiag(\mathbf{Q}_{1},\mathbf{Q}_{2},\dots,\mathbf{Q}_{K})
$$
**Solve**
**Information matrix**: 
$$
\boldsymbol{\Lambda} = \mathbf{C}^T\mathbf{R}^{-1}\mathbf{C} + \mathbf{A}^T\mathbf{Q}^{-1}\mathbf{A}
$$
**Solution**: 
$$
\boxed{\hat{\mathbf{x}} = \boldsymbol{\Lambda}^{-1}(\mathbf{C}^T\mathbf{R}^{-1}\mathbf{y} - \mathbf{A}^T\mathbf{Q}^{-1}\mathbf{v})}
$$
**Covariance**: 
$$
\boxed{\mathbf{P} = \boldsymbol{\Lambda}^{-1}}
$$

# Compressed Notation
The solution can be rewritten in a nice way if we define
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

>[!error] The matricies here are VERY sparse, so its important to optimize for that.

This also cannot be done online because it requires future measurements.