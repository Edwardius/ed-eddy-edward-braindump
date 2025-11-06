Its a matrix of first-order partial derivatives. Used in back propagation.

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[0.5em]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[0.5em]
\vdots & \vdots & \ddots & \vdots \\[0.5em]
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

### From First Principles
$$
\mathbf{J}\triangleq \lim_{ n \to 0 } \frac{f(\mathbf{x}+\mathbf{h})-f(\mathbf{x})}{\mathbf{h}} \in \mathbb{R}^{n\times m}
$$