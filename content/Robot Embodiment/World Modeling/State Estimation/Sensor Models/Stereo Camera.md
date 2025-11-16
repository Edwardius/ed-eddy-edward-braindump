![[Pasted image 20251115195615.png]]
We express everything in a stereo camera with respect to the coordinate frame at the midpoint (known as the *midpoint model*).
$$
\boldsymbol{\rho}=\mathbf{r}_{s}^{ps}=\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
$$
The model for the left and right camera are as follows
$$
\begin{bmatrix}
u_{l} \\
v_{l}
\end{bmatrix}=\mathbf{P}\mathbf{K} \frac{1}{z}\begin{bmatrix}
x+ \frac{b}{2} \\
y \\
z
\end{bmatrix} \quad \begin{bmatrix}
u_{r} \\
v_{r}
\end{bmatrix}=\mathbf{P}\mathbf{K} \frac{1}{z}\begin{bmatrix}
x- \frac{b}{2} \\
y \\
z
\end{bmatrix}
$$
Assuming that the two cameras have the same intrinsic parameters

Stacking the two we get
$$
\begin{bmatrix}
u_{l} \\
v_{l} \\
u_{r} \\
v_{r}
\end{bmatrix}=\mathbf{s}(\boldsymbol{\rho})=\underbrace{ \begin{bmatrix}
f_{u} & 0 & c_{u} & f_{u} \frac{b}{2} \\
0 & f_{v} & c_{v} & 0 \\
f_{u} & 0 & c_{u} & -f_{u} \frac{b}{2 }\\
0 & f_{v} & c_{v} & 0 \\
\end{bmatrix} }_{ \mathbf{M} } \frac{1}{z} \begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$

> [!info] You can also model the stereo camera with respect to the Left or right frames as well.

# Left Model
The camera model becomes
$$
\begin{bmatrix}
u_{l} \\
v_{l} \\
u_{r} \\
v_{r}
\end{bmatrix}=\underbrace{ \begin{bmatrix}
f_{u} & 0 & c_{u} & 0 \\
0 & f_{v} & c_{v} & 0 \\
f_{u} & 0 & c_{u} & -f_{u} b\\
0 & f_{v} & c_{v} & 0 \\
\end{bmatrix} }_{ \mathbf{M} } \frac{1}{z} \begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$
The thing about stereo cameras is that we know the distance between the two cameras and the intrinsics of both. Because they lie in the same plane (only offset by b), we can formulate a relationship between a Point , $P$ z value and **disparity**
$$
d=u_{l}-u_{r}=\frac{1}{z}f_{u}b
$$
but keep in mind that we dont know z, and we have to usually guess using correspondance.

$$
\begin{bmatrix}
u_{l} \\
v_{l} \\
d
\end{bmatrix}=\mathbf{s}(\boldsymbol{\rho})=\underbrace{ \begin{bmatrix}
f_{u} & 0 & c_{u} & 0 \\
0 & f_{v} & c_{v} & 0 \\
0 & 0 & 0& f_{u} b\\
\end{bmatrix} }_{ \mathbf{M} } \frac{1}{z} \begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$
This sensor model is **just telling us how disparity relates to the position of the point**.
![[Pasted image 20251115202345.png]]
(left, disparity if we know the geometry we are looking at)
(right, disparity that we guess from some form of correspondence algorithm)