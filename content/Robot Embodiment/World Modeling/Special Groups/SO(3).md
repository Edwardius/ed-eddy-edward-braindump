Also known as the Special Orthogonal Group. This is also a [[Lie Group]].

**Definition**: Special Orthogonal group in 3D 
$$
\text{SO}(3) = {R \in \mathbb{R}^{3 \times 3} : R^T R = I, \det(R) = 1}
$$
**Group operation**: Matrix multiplication 
$$
R_1, R_2 \in \text{SO}(3) \implies R_1 R_2 \in \text{SO}(3)
$$

**Identity**: $I_{3 \times 3}$

## The Lie Algebra $\mathfrak{so}(3)$
**Definition**: Skew-symmetric 3×3 matrices 
$$
\mathfrak{so}(3) = {\omega^\wedge \in \mathbb{R}^{3 \times 3} : (\omega^\wedge)^T = -\omega^\wedge}
$$

**"Hat" operator** ${}^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$: Maps vectors to skew-symmetric matrices 
$$
\omega = \begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{bmatrix} \quad \xrightarrow{^\wedge} \quad \omega^\wedge = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}
$$

**"Vee" operator** ${}^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$: Inverse of hat (extracts the vector)

**Physical meaning**: $\omega$ is the angular velocity vector (axis-angle representation)

- Direction of $\omega$: axis of rotation
- Magnitude $|\omega|$: angle of rotation (in radians)

**Lie bracket**: $[\omega_1^\wedge, \omega_2^\wedge] = \omega_1^\wedge \omega_2^\wedge - \omega_2^\wedge \omega_1^\wedge = (\omega_1 \times \omega_2)^\wedge$

## Exponential Map: $\mathfrak{so}$(3) → SO(3)
**Rodrigues' Formula**: 
$$
\exp(\omega^\wedge) = I + \frac{\sin(\theta)}{\theta}\omega^\wedge + \frac{1-\cos(\theta)}{\theta^2}(\omega^\wedge)^2
$$
where $\theta = |\omega|$ is the rotation angle.

**Special cases**:
- If $\theta = 0$: $\exp(0) = I$
- Small angles: $\exp(\omega^\wedge) \approx I + \omega^\wedge$ (first-order approximation)

**Alternative form** (axis-angle): 
$$
\exp(\omega^\wedge) = I + \sin(\theta)\hat{u}^\wedge + (1-\cos(\theta))(\hat{u}^\wedge)^2
$$ where $\hat{u} = \omega/\theta$ is the unit axis.

## Logarithm Map: SO(3) → so(3)
$$
\log(R) = \frac{\theta}{2\sin(\theta)}(R - R^T)
$$
where $\theta = \arccos\left(\frac{\text{trace}(R) - 1}{2}\right)$

**Extract the vector**: 
$$
\omega = \log(R)^\vee = \frac{\theta}{2\sin(\theta)}\begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix}
$$

**Special case**: If $\theta = 0$ (R = I), then $\omega = 0$

#worldModeling