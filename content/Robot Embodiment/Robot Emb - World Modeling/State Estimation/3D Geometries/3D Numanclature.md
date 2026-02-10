We have a **reference frame** denoted as
$$
\vec{\mathbf{\mathcal{F}}_{i}}=\begin{bmatrix}
\vec{1}_{1,i} \\
\vec{1}_{2,i} \\
\vec{1}_{3,i}
\end{bmatrix}
$$
A **vector** in that reference frame is denoted as
$$
\vec{r_{i}}=\begin{bmatrix}
r_{1,i} & r_{2,i} & r_{3,i}
\end{bmatrix}\vec{\mathbf{\mathcal{F}}_{i}} \quad \vec{r_{i}}=\mathbf{r}^{T}\vec{\mathbf{\mathcal{F}}_{i}}=\vec{\mathbf{\mathcal{F}}_{i}}^{T}\mathbf{r}
$$
Axes 1, 2, 3 are arbitrarily names, but **they are orthogonal**.

# Cross Product
The cross product of two vectors in the same reference frame is given as
$$
\vec{r} \times \vec{s} = [r_1 \quad r_2 \quad r_3] \begin{bmatrix} \vec{1}_1 \times \vec{1}_1 & \vec{1}_1 \times \vec{1}_2 & \vec{1}_1 \times \vec{1}_3 \\ \vec{1}_2 \times \vec{1}_1 & \vec{1}_2 \times \vec{1}_2 & \vec{1}_2 \times \vec{1}_3 \\ \vec{1}_3 \times \vec{1}_1 & \vec{1}_3 \times \vec{1}_2 & \vec{1}_3 \times \vec{1}_3 \end{bmatrix} \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix}
$$
$$
= [r_1 \quad r_2 \quad r_3] \begin{bmatrix} 0 & \vec{1}_3 & -\vec{1}_2 \\ -\vec{1}_3 & 0 & \vec{1}_1 \\ \vec{1}_2 & -\vec{1}_1 & 0 \end{bmatrix} \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix}
$$
$$
= [\vec{1}_1, \vec{1}_2, \vec{1}_3] \begin{bmatrix} 0 & -r_3 & r_2 \\ r_3 & 0 & -r_1 \\ -r_2 & r_1 & 0 \end{bmatrix} \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix}
$$
$$
= \vec{\mathbf{\mathcal{F}}_{i}}^{T} \mathbf{r}_1^\times \mathbf{s}_1
$$
Where we express r as a *skew-symmetric matrix*
$$\mathbf{r}_1^\times = \begin{bmatrix} 0 & -r_3 & r_2 \\ r_3 & 0 & -r_1 \\ -r_2 & r_1 & 0 \end{bmatrix}$$
>[!error] To adapt to Lie Theory, the $\mathbf{r}^{\times}$ is actually denoted as $\mathbf{r}^{\wedge}$

Relevant for later (also [[SO(3)]])