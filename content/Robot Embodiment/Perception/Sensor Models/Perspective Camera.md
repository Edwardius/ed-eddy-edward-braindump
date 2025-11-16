A perspective camera is just an idealized camera model used for our understanding of computer vision. For this, we will be using the **frontal projection model** ([which is a bit different from what you learned awhile ago](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf). Specifically that the projected plane is on the same side as the object.)

![[Pasted image 20251115185115.png]]

# TLDR
Given we have points in world frame $\vec{r}_{pw}=\begin{bmatrix}x_{w} \\ y_{w} \\ z_{w}\end{bmatrix}$ and the transform between the camera and the world frame is given by
$$
\mathbf{T}_{sw}=\begin{bmatrix}
\mathbf{C}_{sw} & \mathbf{t}_{sw}
\end{bmatrix}
$$
The pinhole camera model is defined as
$$
z_{s}\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=\mathbf{K}\mathbf{T}_{sw}\begin{bmatrix}
\vec{r}_{pw} \\
1
\end{bmatrix}
$$
Which expanded looks like
$$
z_{s}\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=\begin{bmatrix}
\alpha & \gamma & c_{u} \\
0 & \beta & c_{v} \\
0  & 0 & 1
\end{bmatrix}\begin{bmatrix}
\mathbf{C}_{sw} & \mathbf{t}_{sw}
\end{bmatrix}\begin{bmatrix}
x_{w} \\ y_{w} \\ z_{w} \\
1
\end{bmatrix}
$$
- $\mathbf{K}$ is our camera **intrinsic matrix**
	- $\alpha$ is our horizontal focal length accounting for pixel size $\alpha=f_{u}l$
	- $\beta$ is our vertical focal length accounting for pixel size $\beta=f_{v}$
	- $\gamma$ is our **skew coefficient** (usually 0)
	- $c_{u}$ is the horizontal offset to center the image plane with the camera frame
	- $c_{v}$ is the vertical offset to center the image place with the camera frame
- $\mathbf{T}$ is our camera **extrinsic matrix** (chopped transformation matrix)
	- $\mathbf{C}$ is a valid rotation matrix
	- $\mathbf{t}$ is our transformation vector
- $z_{s}$ is the depth of the point in camera frame (must be divided out at the end to get our $u$ and $v$). The act of dividing out the depth is know as **perspective projection**
# Accounting for Distortion
![[Pasted image 20251116140014.png]]
Distortion is a non-linear function, and there are number of different models for distortion.
$$
D(\cdot):\begin{bmatrix}
x_{n} \\
y_{n}
\end{bmatrix} \to \begin{bmatrix}
x_{d} \\
y_{d}
\end{bmatrix}
$$
## Brown-Conrady Distortion Model
AKA **Plumb Bob Model**
$$
x_{d}=x_{n} \underbrace{ \frac{1+k_{1}r^{2}+k_{2}r^{4}+k_{3}r^{6}}{1+k_{4}r^{2}+k_{5}r^{4}+k_{6}r^{6}} }_{ \text{radial distortion} }+\underbrace{ 2p_{1}x_{n}y_{n}+p_{2}(r^{2}+2x_{n}^{2}) }_{ \text{tangential distortion} }+\underbrace{ s_{1}r^{2}+s_{2}r^{4} }_{ \substack{\text{thin prism}
\\ \text{distortion}\\ \text{(misaligned} \\ \text{lenses)}} }
$$
$$
y_{d}=y_{n} \frac{1+k_{1}r^{2}+k_{2}r^{4}+k_{3}r^{6}}{1+k_{4}r^{2}+k_{5}r^{4}+k_{6}r^{6}}+p_{1}(r^{2}+2y_{n}^{2})+2p_{2}x_{n}y_{n}+s_{3}r^{2}+s_{4}r^{4}
$$
where
$$
r^{2}=x_{n}^{2}+y_{n}^{2}
$$
>[!caution] This is the distortion model used by OpenCV

**However,** most of the time OpenCV uses the standard 5-parameter, simplified model for most calibrations.
$$
x_{d}=x_{n}(1+k_{1}r^{2}+k_{2}r^{4}+k_{3}r^{6})+2p_{1}x_{n}y_{n}+p_{2}(r^{2}+2x_{n}^{2})
$$
$$
y_{d}=y_{n} (1+k_{1}r^{2}+k_{2}r^{4}+k_{3}r^{6})+p_{1}(r^{2}+2y_{n}^{2})+2p_{2}x_{n}y_{n}
$$
## Adding Distortion into the model
>[!error] We add in the distortion **after the extrinsic and perspective transform**, **before passing through camera intrinsics**

$$
z_{s}\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=\begin{bmatrix}
\alpha & \gamma & c_{u} \\
0 & \beta & c_{v} \\
0  & 0 & 1
\end{bmatrix}\begin{bmatrix}
\mathbf{C}_{sw} & \mathbf{t}_{sw}
\end{bmatrix}\begin{bmatrix}
x_{w} \\ y_{w} \\ z_{w} \\
1
\end{bmatrix}
$$
$$
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=\begin{bmatrix}
\alpha & \gamma & c_{u} \\
0 & \beta & c_{v} \\
0  & 0 & 1
\end{bmatrix}D\left(\frac{\mathbf{T}_{sw}\begin{bmatrix}
\vec{r}_{pw} \\
1
\end{bmatrix}}{z_{s}}\right)
$$
# Explanation
We have our pinhole $S$ and a point in 3D space, $P$. The vector to the point from the pinhole is
$$
\rho=\mathbf{r}_{s}^{ps}=\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
$$
>[!info] important to note that the $\vec{s_{3}}$ axis is normal to the image plane

The projected point $O$ which will end up on the plane is given by a vector $\mathbf{p}$
$$
\mathbf{p}=\begin{bmatrix}
x_{n} \\
y_{n} \\
1
\end{bmatrix}=\begin{bmatrix}
\frac{x}{z} \\
\frac{y}{z} \\
1
\end{bmatrix}
$$
It is homogeneous to work with matrix calculations. This is called a **normalized image coordinate**

# Essential Matrix
![[Pasted image 20251115185844.png]]
If the same point is collected by the same camera after a transformation, the two observations of the same point are related by
$$
\mathbf{p}_{a}^{T}\mathbf{E}_{ab}\mathbf{p}_{b}=0
$$
Where $\mathbf{E}$ is called the **essential matrix**. 
$$
\mathbf{E}_{ab}=\mathbf{C}_{ba}^{T}\mathbf{r}_{b}^{ab\wedge}
$$
And its related to the pose of the camera
$$
\mathbf{T}_{ba}=\begin{bmatrix}
\mathbf{C}_{ba} & \mathbf{r}_{b}^{ab} \\
\mathbf{0}^{T} & 1
\end{bmatrix}
$$
# Lens Distortion
It exists, and we need to characterize it and deal with it, This affects how close a real camera is to our idealized model. But once it is characterized, we can run an undistortion procedure to get the camera image to something we can actually use.

# Intrinsic Parameters
![[Pasted image 20251115190939.png]]
We were assuming right before this that the focal length is 1, so we actually need to deal with that (as well as map to pixel coordinates which start from the top left of the image)
$$
\mathbf{q}=\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=\mathbf{K}\mathbf{p}=\begin{bmatrix}
f_{u} & 0 & c_{u} \\
0 & f_{v} & c_{v} \\
0  & 0 & 1
\end{bmatrix}\begin{bmatrix}
x_{n} \\
y_{n} \\
1
\end{bmatrix}
$$
These have to be determined through **camera calibration**

# Fundamental Matrix
Similar to the Essential Matrix, but defined between two different cameras. Lets say we have two different cameras
$$
\mathbf{q}_{a}=\mathbf{K}_{a}\mathbf{p}_{a}
$$
$$
\mathbf{q}_{b}=\mathbf{K}_{b}\mathbf{p}_{b}
$$
The fundamental matrix, $\mathbf{F}_{ab}$ exists such that
$$
\mathbf{q}_{a}^{T}\mathbf{F}_{ab}\mathbf{q}_{b}=0 \quad \text{where} \quad \mathbf{F}_{ab}=\mathbf{K}_{a}^{-T}\mathbf{E}_{ab}\mathbf{K}_{b}^{-1}
$$
Reasoning
$$
\mathbf{q}_{a}^{T}\mathbf{F}_{ab}\mathbf{q}_{b}=\mathbf{p}_{a}^{T}\underbrace{ \mathbf{K}_{a}^{T}\mathbf{K}_{a}^{-T} }_{ 1 }\mathbf{E}_{ab}\underbrace{ \mathbf{K}_{b}^{-1}\mathbf{K}_{b} }_{ 1 }\mathbf{p}_{b}=\mathbf{p}_{a}^{T}\mathbf{E}_{ab}\mathbf{p}_{b}=0
$$
The constraint associated with the fundamental matrix is also called the **epipolar constraint**
![[Pasted image 20251115192233.png]]

# Homography
If an observed point is on a plane of known geometry, it is possible to work out what the point will look like on another camera of a known pose change.  This is called **homography**.

![[Pasted image 20251115194350.png]]

![[Pasted image 20251115192830.png]]

Like before, we have two cameras
$$
\mathbf{q}_{a}=\mathbf{K}_{a}\mathbf{p}_{a}=\mathbf{K}_{a} \frac{1}{z_{a}}\boldsymbol{\rho}_{a}
$$
$$
\mathbf{q}_{b}=\mathbf{K}_{b}\mathbf{p}_{b}=\mathbf{K}_{b} \frac{1}{z_{b}}\boldsymbol{\rho}_{b}
$$
Say we know the equation of the Plane expressed in both camera frames to be
$$
\{ \mathbf{n}_{a},d_{a} \} \text{ and } \{ \mathbf{n}_{b},\mathbf{d}_{b} \}
$$
This implies that
$$
\mathbf{n}_{a}^{T}\boldsymbol{\rho}_{a}+d_{a}=0
$$
$$
\mathbf{n}_{b}^{T}\boldsymbol{\rho}_{b}+d_{b}=0
$$
Substituting our equations for $\rho$
$$
z_{i}\mathbf{n}_{i}^{T}K_{i}^{-1}\mathbf{q}_{i}+d_{i}=0\quad i=a,b
$$
$$
\text{or }z_{i}=- \frac{d_{i}}{\mathbf{n}_{i}^{T}\mathbf{K}_{i}^{-1}\mathbf{q}_{i}}
$$
This implies that we can write the coordinates of $P$ with respect to any camera frame as
$$
\mathbf{q}_{i}=\mathbf{K}_{i} \frac{1}{z_{i}}\boldsymbol{\rho}_{i}
$$
$$
\boldsymbol{\rho}_{i}=z_{i}\mathbf{K}_{i}^{-1}\mathbf{q}_{i}
$$
$$

\boldsymbol{\rho}_{i}=- \frac{d_{i}}{\mathbf{n}_{i}^{T}\mathbf{K}_{i}^{-1}\mathbf{q}_{i}}\mathbf{K}_{i}^{-1}\mathbf{q}_{i}
$$
Extending this to our coodinates in the image plane, we get that
$$
\mathbf{q}_{b}=\mathbf{K}_{b}\mathbf{H}_{ba}\mathbf{K}_{a}^{-1}\mathbf{q}_{a} \quad \underbrace{ \mathbf{H}_{ba}=\frac{z_{a}}{z_{b}}\mathbf{C}_{ba}\left( \mathbf{1}+ \frac{1}{d_{a}}\mathbf{r}_{a}^{ba}\mathbf{n}_{a}^{T} \right) }_{ \text{Homography Matrix } }
$$
The **Homography Matrix** lets us determine the how a point in one image plane is gonna look like in another plane. Given that we know the geometry of the point

The Homography matrix is invertible
$$
\mathbf{H}_{ba}^{-1}=\mathbf{H}_{ab}
$$
