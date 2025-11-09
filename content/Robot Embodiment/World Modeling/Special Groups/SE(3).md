#stateEstimation 

> [!error] This shit is fucking dumb, I can't believe for the longest time I never decided to go further than a working knowledge of state estimation, and now I'm being hit with Lie Theory which is fucking weird.

$$
SE(3) = \left\{ T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4} \,\bigg|\, R \in SO(3), \, \mathbf{t} \in \mathbb{R}^3 \right\}
$$
$$
SO(3) = \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R) = 1 \}
$$
$$
SE(3) \cong SO(3) \times \mathbb{R}^3
$$
# SE(3): The Special Euclidean Group

**SE(3)** is the group of rigid body transformations in 3D space, combining rotations (defined by [[SO(3)]]) and translations.

## Group Structure
An element of SE(3) is represented as: 
$$
T = \begin{bmatrix} R & \mathbf{t} \\ 0^T & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$
where $R \in SO(3)$ is a rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is a translation vector. The group operation is matrix multiplication.

## Lie Algebra $\mathfrak{se}$(3)
The tangent space at the identity forms the **Lie algebra se(3)**, consisting of 4×4 matrices: $$\xi^\wedge = \begin{bmatrix} [\omega]_\times & \mathbf{v} \\ 0^T & 0 \end{bmatrix}$$
where:
- $[\omega]_\times \in so(3)$ is a skew-symmetric matrix (angular velocity)
- $\mathbf{v} \in \mathbb{R}^3$ is linear velocity

The 6D vector form is $\xi = \begin{bmatrix} \mathbf{v} \ \omega \end{bmatrix} \in \mathbb{R}^6$ (or sometimes ordered as $[\omega, \mathbf{v}]$).

> [!info] We often collapse the Lie Algebra space down to a more comprehensive vector space, and do processing on that.

## Exponential Map: se(3) → SE(3)
The exponential map takes Lie algebra elements to the group: 
$$
\exp(\xi^\wedge) = \begin{bmatrix} \exp([\omega]_\times) & J\mathbf{v} \\ 0^T & 1 \end{bmatrix}
$$

where:

- $\exp([\omega]_\times)$ is the SO(3) exponential (Rodrigues' formula)
- $J$ is the left Jacobian of SO(3): $J = I + \frac{1-\cos\theta}{\theta^2}[\omega]_\times + \frac{\theta - \sin\theta}{\theta^3}[\omega]_\times^2$
- $\theta = |\omega|$

For small angles, $J \approx I + \frac{1}{2}[\omega]_\times$.
## Logarithm Map: SE(3) → se(3)
The logarithm is the inverse operation: 
$$
\log(T) = \begin{bmatrix} [\omega]_\times & J^{-1}\mathbf{t} \\ 0^T & 0 \end{bmatrix}
$$

where $\omega$ is recovered from $\log(R)$ and: 
$$
J^{-1} = I - \frac{1}{2}[\omega]_\times + \left(\frac{1}{\theta^2} - \frac{1+\cos\theta}{2\theta\sin\theta}\right)[\omega]_\times^2
$$
#worldModeling