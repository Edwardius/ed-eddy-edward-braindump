In [[Gauss-Newton Method]], we make the assumption that we are **already near a local minima**. This is because of our assumption about the Hessian approximation. Which assumes that we are already near the optimum.

$$
\left(\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right) + \lambda\mathbf{D}\right) \delta\mathbf{x}^* = -\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \mathbf{u}(\mathbf{x}_{\text{op}})
$$

where $\mathbf{D}$ is a positive diagonal matrix. When $\mathbf{D} = \mathbf{1}$, we can see that as $\lambda \geq 0$ becomes very big, the Hessian is relatively small, and we have

$$
\delta\mathbf{x}^* \approx -\frac{1}{\lambda}\left(\frac{\partial \mathbf{u}(\mathbf{x})}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{\text{op}}}\right)^T \mathbf{u}(\mathbf{x}_{\text{op}})
$$

which corresponds to a very small step in the direction of steepest descent. When $\lambda=0$ we recover [[Gauss-Newton Method]]

>[!error] By controlling $\lambda$, we can decide when we want to do regular gradient descent and rapid Gauss-Newton optimization

When our initial estimate is very far from the optimum, we can slow down our optimizer to do gradient descent until we get near a optimum and rapidly converge to it with Gauss-Newton.

>[!error] LARGE $\lambda$ MEANS SMALLER STEP, SMALL $\lambda$ MEANS BIGGER STEPS / MORE GAUSS-NEWTONY STEPS

