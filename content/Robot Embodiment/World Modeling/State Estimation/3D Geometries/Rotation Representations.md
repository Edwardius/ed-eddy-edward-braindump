Given a reference frame or vector defined as from [[3D Numanclature]]
$$
\vec{\mathbf{\mathcal{F}}_{i}}=\begin{bmatrix}
\vec{1}_{1,i} \\
\vec{1}_{2,i} \\
\vec{1}_{3,i}
\end{bmatrix} \quad
\vec{r_{i}}=\begin{bmatrix}
\vec{1}_{1,i}r_{1} \\
\vec{1}_{2,i}r_{2} \\
\vec{1}_{3,i}r_{3}
\end{bmatrix}
$$
# Rotation Matrix
$$
\mathbf{C}\in \mathbb{R}^{3\times3}
$$
Able to perform a rotation on a vector $\mathbf{r}$ such that
$$
\mathbf{r}_{1}=\mathbf{C}_{21}^{-1}\mathbf{r}_{2}=\mathbf{C}_{12}\mathbf{r}_{2} \quad \mathbf{r}_{2}=\mathbf{C}_{21}\mathbf{r}_{1}
$$
$$
\mathbf{r}_{3}=\mathbf{C}_{32}\mathbf{r}_{2}=\mathbf{C}_{32}\mathbf{C}_{21}\mathbf{r}_{1} \;\; \text{therefore} \;\; \mathbf{C}_{31}=\mathbf{C}_{32}\mathbf{C}_{21}
$$
So you can chain them!

# Euler Angles
**Two types** of euler representations.
## 3($\varphi$)-1($\gamma$)-3($\theta$)
Consists of a rotation about the 3-axis, and then the 1-axis, and then the transformed 3 axis.
![[Pasted image 20251115140450.png]]

$\theta$ is **spin** angle
$\gamma$ is **nutation** angle
$\varphi$ is **precession** angle

To convert it to a rotation matrix:
$$
\mathbf{C}_{21}(\theta, \gamma, \psi) = \mathbf{C}_{2J}\mathbf{C}_{J1}\mathbf{C}_{11}
$$
$$
= \mathbf{C}_3(\theta)\mathbf{C}_1(\gamma)\mathbf{C}_3(\psi)
$$
$$
= \begin{bmatrix} 
c_\theta c_\psi - s_\theta c_\gamma s_\psi & s_\theta c_\theta + c_\gamma s_\theta c_\psi & s_\gamma s_\psi \\
-c_\psi s_\theta - c_\theta c_\gamma s_\psi & -s_\theta s_\psi + c_\theta c_\gamma c_\psi & s_\gamma c_\psi \\
s_\psi s_\gamma & -s_\gamma c_\psi & c_\gamma
\end{bmatrix}
$$

have made the abbreviations $s = \sin$, $c = \cos$
# 1($\theta_{1}$)-2($\theta_{2}$)-3($\theta_{3}$)
Consists of a rotation in the 1-axis, then the 2-axis, and finally the 3-axis

To convert to rotation matrix
$$
\mathbf{C}_{21}(\theta_3, \theta_2, \theta_1) = \mathbf{C}_3(\theta_3)\mathbf{C}_2(\theta_2)\mathbf{C}_1(\theta_1)
$$
$$
= \begin{bmatrix} 
c_2c_3 & c_1s_3 + s_1s_2c_3 & s_1s_3 - c_1s_2c_3 \\
-c_2s_3 & c_1c_3 - s_1s_2s_3 & s_1c_3 + c_1s_2s_3 \\
s_2 & -s_1c_2 & c_1c_2
\end{bmatrix}
$$
>[!caution] All Euler sequences have singularities. That is, a point in which you lose one degree of freedom because two of the axes of rotation are the same. This is known a gimbal lock (or in more math terms you've reached a singularity)

The term gimbal lock comes from the fact that a gimbal measurement device (like an IMU) is rotated in space such that two of the gimbal rings line up. They seemingly get stuck together because of physics and all of the sudden our IMU loses a degree of freedom to measure with.

https://www.youtube.com/watch?v=oj7v3MXJL3M
![[Pasted image 20251115145122.png]]
**At this point in time of the gimbal, two axes line up and you end up with infinite gimbal states that could give you the rotation of the spinning core.** This is bad because our gimbal IMU reading starts to go crazy.

In rotational geometry, this is referred to as reaching a **singularity**. For 1-2-3, this happens when $\theta_{2}=\frac{\pi}{2}$

When trying to convert into a rotation matrix, we get:
$$
\mathbf{C}_{21}(\theta_3, \frac{\pi}{2}, \theta_1) = \begin{bmatrix} 
0 & \sin(\theta_1 + \theta_3) & -\cos(\theta_1 + \theta_3) \\
0 & \cos(\theta_1 + \theta_3) & \sin(\theta_1 + \theta_3) \\
1 & 0 & 0
\end{bmatrix}
$$
As a result, **when we want to recover euler angles from a rotation matrix, we no longer have a unique solution for $\theta_{1},\theta_{3}$**

>[!error] All Euler Angles have a singularity.

# Infinitesimal Rotations 
For small 1-2-3 transformations, we can approximate the rotation matrix as
$$
\mathbf{C}_{21}=\begin{bmatrix}
1 & \theta_{3} & -\theta_{2} \\
-\theta_{3} & 1 & \theta_{1} \\
\theta_{2} & -\theta_{1} & 1
\end{bmatrix}=\mathbf{1}-\theta^{\times}
$$

# Euler Parameters
From *Euler's Rotation Theorem* , the most general motion of a rigid body with one point fixed is a rotation about an axis through that point.

![[Pasted image 20251115145656.png]]

For our case, we define that axis as a **unit vector**
$$
\mathbf{a}=\begin{bmatrix}
a_{1} & a_{2} & a_{3}
\end{bmatrix}^{T} \quad \mathbf{a}^{T}\mathbf{a}=1
$$
The **angle of rotation** is defined as $\phi$

**A rotation matrix is given by**
$$
\mathbf{C}_{21}=\cos \phi \mathbf{1}+(1-\cos \phi)\mathbf{a}\mathbf{a}^{T}-\sin \phi \mathbf{a}^{\times}
$$
**Euler parameters** are defined from this:
$$
\eta=\cos \frac{\phi}{2}\quad\epsilon=\mathbf{a}\sin \frac{\phi}{2}=\begin{bmatrix}
a_{1}\sin\left( \frac{\phi}{2} \right) \\
a_{2}\sin\left( \frac{\phi}{2} \right) \\
a_{3}\sin\left( \frac{\phi}{2} \right)
\end{bmatrix}=\begin{bmatrix}
\epsilon_{1} \\
\epsilon_{2} \\
\epsilon_{3}
\end{bmatrix}
$$
These parameters are **not** independant because
$$
\eta^{2}+\epsilon_{1}^{2}+\epsilon_{2}^{2}+\epsilon_{3}^{2}=1
$$
Stacked as $\mathbf{q}=\begin{bmatrix}\epsilon \\ \eta\end{bmatrix}$ gives us **quarternions**
# Quarternions
Follows from [Euler Parameters](#Euler Parameters) 
$$
\mathbf{q}=\begin{bmatrix}
\epsilon \\
\eta
\end{bmatrix}
$$
There are special kinds of operators on quarternions called the **left-hand** and **right hand** compound operators.
$$
\mathbf{q}^{+}=\begin{bmatrix}
\eta \mathbf{1}-\epsilon^{\times} & \epsilon \\
-\epsilon^{T} & \eta 
\end{bmatrix}\quad\mathbf{q}^{\oplus}=\begin{bmatrix}
\eta \mathbf{1}+\epsilon^{\times} & \epsilon \\
-\epsilon^{T} & \eta
\end{bmatrix}
$$
And the inverse operator is defined explicitly for quarternions to be
$$
\mathbf{q}^{-1}=\begin{bmatrix}
-\epsilon \\
\eta
\end{bmatrix}
$$
Some useful identities with these operators...
![[Pasted image 20251115151742.png]]

**Quarternions form a non-commutative group** under both the $\oplus+$ operators. With the identity element of the group being
$$
\iota=\begin{bmatrix}
0 \\
0 \\
0 \\
1
\end{bmatrix}\quad
\iota^{+}=\iota^{\oplus}=\mathbf{1}
$$

## Rotating with a Quarternion
given a point (in homogeneous form)
$$
\mathbf{v}=\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$
We can rotate it using the quarterion, $\mathbf{q}$, by
$$
\mathbf{u}=\mathbf{q}^{+}\mathbf{v}^{+}\mathbf{q}^{-1}=\mathbf{q}^{+}\mathbf{q}^{-1\oplus}\mathbf{v}=\mathbf{R}\mathbf{v}
$$
If just so follows that we end up with [[SO(3)]] by getting $\mathbf{R}$
$$
\mathbf{R}=\mathbf{q}^{+}\mathbf{q}^{-1\oplus}=\mathbf{q}^{-1\oplus}\mathbf{q}^{+}=\mathbf{q}^{\oplus^{T}}\mathbf{q}^{+}=\begin{bmatrix}
\mathbf{C} & \mathbf{0} \\
\mathbf{0^{T}} & 1
\end{bmatrix}
$$
# Gibbs Vector
$$
\mathbf{g}=\mathbf{a}\tan \frac{\phi}{2}
$$
Has a singularity at $\frac{\pi}{2}$. 

Less common, but worth noting.