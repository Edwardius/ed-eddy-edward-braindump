Similar to [[LG Problem Statement]] we define a similar discrete-time, time-invariant model. Except this time it's non-linear and non-Gaussian. This is known as a NLNG Model.

It can be characterized as:
$$
\text{motion model:}\;\; \mathbf{x}_{k}=\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{v}_{k}, \mathbf{w}_{k}), \;\;k=1\dots K
$$
$$
\text{observation model:}\;\; \mathbf{y}_{k}=\mathbf{g}(\mathbf{x}_{k},\mathbf{n}_{k}), \;\; k=0\dots K
$$
Where $\mathbf{f}(\cdot)$ is a **non-linear motion model** and $\mathbf{g}(\cdot)$ is a **non-linear observation model**.

This system is a [[Markovian]].


