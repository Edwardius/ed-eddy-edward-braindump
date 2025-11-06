Its a fundamental concept in mathematics and physics that lets us study the local, linear behaviour of a corresponding non-linear [[Lie Group]] at its identity.

This is because as you zoom into a local area on a Lie Group, it becomes linear in nature. **THE LINEAR SPACE ABOUT THE IDENTITY IS CALLED THE LIE ALGEBRA**.


![[Pasted image 20251105211125.png]]
> [!info] Representation of the relation between the Lie group and the Lie algebra. The Lie algebra (red plane) is the tangent space to the Lie group's manifold (here represented as a blue sphere) at the identity. Through the exponential map, each straight path through the origin on the Lie algebra produces a path over the manifold which runs along the respective geodesic. Conversely, each element of the group has an equivalent in the Lie algebra. This relation is so profound that (nearly) all operations in the group, which is curved and nonlinear, have an exact equivalent in the Lie algebra, which is a linear vector space. Though the sphere in R^3 is not a Lie group (we just use it as a representation that can be drawn on paper), that in R^4 is, and describes the group of unit quaternions.

# Exponential and Logarithm

The conversions between the lie algebra and lie group is known as exponential and log.
- **exp:** Lie Algebra -> Lie Group
- **log:** Lie Group -> Lie Algebra

## Isomorphisms
When we derive the Lie Algebra, we can end up in a space thats a bit weird to work with (ie. in $S^{1}$ we end up in the space $i\mathbb{R}$). As a result, we'd like to do a small helper operation that lets us directly handle Lie Algebra ($i\mathbb{R}$, $\mathbb{R}^{3\times 3}$) in a Cartesian space ($\mathbb{R}$, $\mathbb{R}^{6}$). These are what we define as the **Hat** and **Vee** Operators

![[Pasted image 20251106091510.png]]

![[Pasted image 20251106091018.png]]

When we do a **Vee** operation on the Lie Algebra, we end up with a nice Cartesian space to do think in vectors. The corresponding operations to go from this space to the Lie Group is the Exponential and Logarithm (just capitalized).
# Plus Minus Operators
There are special operators that we define so that we can equate addition in the Lie Algebra vector space to the resultant associative Lie Group
$$
\mathcal{X}\oplus \mathcal{w} \triangleq \mathcal{X}*Exp(\mathcal{w})
$$
$\triangleq$ means **defined as**, and remember $*$ is the operator of the Lie Group

![[Pasted image 20251106090527.png]]

Likewise, there is an equivalent minus operator in Lie Algebra
$$
\mathcal{Y} \ominus \mathcal{X} \triangleq Log(\mathcal{X}^{-1}*\mathcal{Y})
$$
# Adjoint Matrix
From the special plus operator:

![[Pasted image 20251106092019.png]]
- This means that we have a way to map one vector in Lie Algebra to another.
# Calculus on the Lie Group

> [!error] We can express the operation of two elements in the Lie Group as the operation on two Cartesian vectors in the Lie Algebra! As a result, we can define our own ideas of **Jacobian** and **Covariance** in this space.

![[Pasted image 20251106092301.png]]
# Jacobians in Lie Group
**In vector space, we define the [[Jacobian]] from first principles as:**
$$
\mathbf{J}\triangleq \lim_{ h \to 0 } \frac{f(\mathbf{x}+\mathbf{h})-f(\mathbf{x})}{\mathbf{h}} \in \mathbb{R}^{n\times m}
$$
**You can define something similar for Jacobians in Lie Algebra:**
$$
\mathbf{J}\triangleq\lim_{ \boldsymbol{\tau} \to 0 }\frac{ f(\mathcal{X}\oplus \boldsymbol{\tau}) \ominus f(\mathcal{X}) }{\boldsymbol{\tau}} \in \mathbb{R}^{n\times m}
$$
![[Pasted image 20251106093833.png]]

---

**EXAMPLE** with [[SO(3)]]

Given we have a function $f$ defined as $f:SO(3)\times \mathbb{R}^{3}\to \mathbb{R}^{3};\;(\mathbf{R},\mathbf{p})\mapsto f(\mathbf{R},\mathbf{p})=\mathbf{R}*\mathbf{p}$

$$
\frac{Df}{D\mathbf{R}}=\lim_{ \boldsymbol{\boldsymbol{\theta}} \to 0 }\frac{ f(\mathcal{\mathbf{R}}\oplus \boldsymbol{\theta},\mathbf{\mathbf{p}}) \ominus f(\mathbf{R},\mathbf{\mathbf{p}}) }{\boldsymbol{\boldsymbol{\theta}}}=\lim_{ \theta \to 0 } \frac{(\mathbf{R}\oplus \boldsymbol{\theta})*\mathbf{p}-\mathbf{R}*\mathbf{p}}{\boldsymbol{\theta}}
$$
$$
=\lim_{ \theta \to 0 } \frac{\mathbf{R}*Exp(\boldsymbol{\theta})*\mathbf{p}-\mathbf{R}*\mathbf{p}}{\boldsymbol{\theta}}
$$
For SO(3) we can approximate the Exponential map as:
$$
Exp(\boldsymbol{\theta})=\mathbf{I}+\theta_{\times}
$$
Where $\theta_{\times}$ is the skew symmetric matrix.
$$
\theta_{\times}=\begin{bmatrix}
0 & -\theta_{3} &  \theta_{2} \\
\theta_{3} & 0 & -\theta_{1} \\
-\theta_{2} & \theta_{1} & 0 
\end{bmatrix}
$$
$$
=\lim_{ \theta \to 0 } \frac{\mathbf{R}*(\mathbf{I}+\theta_{\times})*\mathbf{p}-\mathbf{R}*\mathbf{p}}{\boldsymbol{\theta}}
$$
$$
=\lim_{ \theta \to 0 } \frac{\mathbf{R}*\mathbf{I}*\mathbf{p}+\mathbf{R}*\theta_{\times}*\mathbf{p}-\mathbf{R}*\mathbf{p}}{\boldsymbol{\theta}}
$$
$$
=\lim_{ \theta \to 0 } \frac{\mathbf{R}*\theta_{\times}*\mathbf{p}}{\boldsymbol{\theta}}
$$
$$
=\lim_{ \theta \to 0 } \frac{-\mathbf{R}*\mathbf{p}_{\times}*\boldsymbol{\theta}}{\boldsymbol{\theta}}
$$
Why? Well we can expand and see:
$$
\boldsymbol{\theta}_{\times} \cdot \mathbf{p} = \begin{bmatrix} 0 & -\theta_3 & \theta_2 \\ \theta_3 & 0 & -\theta_1 \\ -\theta_2 & \theta_1 & 0 \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \begin{bmatrix} \theta_2 p_3 - \theta_3 p_2 \\ \theta_3 p_1 - \theta_1 p_3 \\ \theta_1 p_2 - \theta_2 p_1 \end{bmatrix}
$$
$$
\mathbf{p}_{\times} \cdot \boldsymbol{\theta} = \begin{bmatrix} 0 & -p_3 & p_2 \\ p_3 & 0 & -p_1 \\ -p_2 & p_1 & 0 \end{bmatrix} \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} p_2 \theta_3 - p_3 \theta_2 \\ p_3 \theta_1 - p_1 \theta_3 \\ p_1 \theta_2 - p_2 \theta_1 \end{bmatrix}
$$
$$
\text{therefore}\;\boldsymbol{\theta}_{\times} \cdot \mathbf{p} = -\mathbf{p}_{\times} \cdot \boldsymbol{\theta}
$$
$$
=-\mathbf{R}*\mathbf{p}_{\times}
$$

---
# Differentiation in Lie Groups
Just a couple of notation definitions and some deductions from those definitions.

![[Pasted image 20251106100302.png]]

# [[Pertubation]] in Lie Groups and Covariance
So moving a little bit LOL.
![[Pasted image 20251106101338.png]]

Lets say we define a pertubation in Lie Algebra.
$$
\mathcal{X}=\mathcal{\bar{X}}\oplus \boldsymbol{\tau} \;\;\text{where}\;\;\boldsymbol{\tau} \sim \mathcal{N}(0,\mathbf{P})
$$Where $\tau$ represents some random vector.

From there, we can see how covariance comes into play:
$$
\mathbf{P}\triangleq \mathbb{E}[\boldsymbol{\tau}*\boldsymbol{\tau}^{T}]
$$
$$
\mathbf{P}\triangleq\mathbb{E}[(\mathcal{X}\ominus\mathcal{\bar{X}})*(\mathcal{X}\ominus\mathcal{\bar{X}})^{T}]
$$
Propagation is pretty easy once you just "think" in Lie Algebra lol.
$$
\mathcal{Y}=f(\mathcal{X})\;\;\mathbf{J}=\frac{D\mathcal{Y}}{D\mathcal{X}} 
$$
$$
\mathbf{P}_{\mathcal{Y}}=\mathbf{J}*\mathbf{P}_{\mathcal{X}}*\mathbf{J}^{T}
$$
# Integration in Lie Groups
![[Pasted image 20251106102003.png]]