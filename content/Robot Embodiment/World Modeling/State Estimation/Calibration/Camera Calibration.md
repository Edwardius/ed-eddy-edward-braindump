# Camera Calibration

We seek to discover the intrinsic parameters of a camera. There are various approaches to do this.

## Setup

I got some pretty shitty images of a checkerboards.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob
```

```python
image_paths = glob.glob('images/*.png')
images = []

for path in image_paths:
  img = cv2.imread(path)
  if img is not None:
    images.append(img)
  else:
    print("No images found, or failed to load")

# Convert BGR to RGB for matplotlib
images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images if img is not None]

# Display in a grid
n_images = len(images_rgb)
cols = 5  # Number of columns
rows = 2

plt.figure(figsize=(15, 3 * rows))  # Adjust figure size

for i, img in enumerate(images_rgb):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

images[0].shape
```
![[Pasted image 20251115220617.png]]
# Calibration Time

Using opencv calibration is pretty easy, there's a built in function

```python
pattern_size = (10, 7)
square_size = 0.0865

# 3D points of checkerboard corners (always the same)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Storage
objpoints = []  # 3D points (same for all images)
imgpoints = []  # 2D points (different per image)

# Process each image
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        objpoints.append(objp)  # Same 3D points
        imgpoints.append(corners)  # Different 2D observations

# Calibrate
h, w = images[0].shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

print(f"Intrinsics K:\n{K}")
print(f"Distortion {dist}")
print(f"Number of extrinsic poses: {len(rvecs)}")  # = number of images
```

**Output:**
```
Intrinsics K: 
[[2.15670900e+03 0.00000000e+00 8.10347489e+02] 
[0.00000000e+00 2.14267203e+03 5.48028224e+02] 
[0.00000000e+00 0.00000000e+00 1.00000000e+00]] 
Distortion:
[[ 0.0210893 -0.68716169 -0.00217461 -0.01650779 1.76950328]] Number of extrinsic poses: 10
```

So I got a pretty shit result, maybe because the checkerboard was warping LOL.

What is it doing under the hood?

# Zhang's Method
1. **Capture images**: Take 10-20 photos of a checkerboard pattern from different angles and distances
2. **Detect corners**: Automatically locate the checkerboard corners in each image
3. **Establish correspondences**: Map 2D image points to 3D world coordinates on the plane
4. **Compute homographies**: Calculate the projective transformation between the plane and each image (This is done with SVD on the linear system produced by our detected corners and correspondances)
5. **Extract parameters**: Use these homographies to solve for camera parameters through closed-form solutions followed by nonlinear optimization

Given our camera model
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
We let the checkboard be the reference frame, so $z_{w}=0$ and our model simplified to

$$
z_s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix} \begin{bmatrix} x_w \\ y_w \\ 1 \end{bmatrix} = H \begin{bmatrix} x_w \\ y_w \\ 1 \end{bmatrix}
$$

From H = λK[r₁ r₂ t], we can write: 
$$
H = [h_1 \quad h_2 \quad h_3] = \lambda K[r_1 \quad r_2 \quad t]
$$
where $\lambda$ is a scaling factor which is left over and we have to deal with

So: **r₁ = λK⁻¹h₁** and **r₂ = λK⁻¹h₂**

Since r₁ and r₂ are orthonormal:

- **r₁ᵀr₂ = 0** → **h₁ᵀK⁻ᵀK⁻¹h₂ = 0**
- **r₁ᵀr₁ = r₂ᵀr₂** → **h₁ᵀK⁻ᵀK⁻¹h₁ = h₂ᵀK⁻ᵀK⁻¹h₂**

Let B = K⁻ᵀK⁻¹. Expanding this symmetric matrix:

$$
B = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

In terms of intrinsics: 
$$
B = \begin{bmatrix} \frac{1}{\alpha^2} & -\frac{\gamma}{\alpha^2\beta} & \frac{\gamma c_v - \beta c_u}{\alpha^2\beta} \\ -\frac{\gamma}{\alpha^2\beta} & \frac{\gamma^2}{\alpha^2\beta^2} + \frac{1}{\beta^2} & -\frac{\gamma(c_v\gamma - \beta c_u)}{\alpha^2\beta^2} - \frac{c_v}{\beta^2} \\ \frac{\gamma c_v - \beta c_u}{\alpha^2\beta} & -\frac{\gamma(c_v\gamma - \beta c_u)}{\alpha^2\beta^2} - \frac{c_v}{\beta^2} & \ldots \end{bmatrix}
$$

We can represent B as a 6D vector: **b = [B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃]ᵀ**

For each homography H with columns h₁, h₂, h₃, define:

$$
v_{ij} = [h_{i1}h_{j1}, \quad h_{i1}h_{j2} + h_{i2}h_{j1}, \quad h_{i2}h_{j2}, \quad h_{i3}h_{j1} + h_{i1}h_{j3}, \quad h_{i3}h_{j2} + h_{i2}h_{j3}, \quad h_{i3}h_{j3}]^T
$$

Then: **hᵢᵀBhⱼ = vᵢⱼᵀb**

The two constraints become: $$\begin{bmatrix} v_{12}^T \\ (v_{11} - v_{22})^T \end{bmatrix} b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
Each image gives 2 equations. With n images:

$$
\underbrace{\begin{bmatrix} v_{12}^T \text{ (img 1)} \\ (v_{11} - v_{22})^T \text{ (img 1)} \\ v_{12}^T \text{ (img 2)} \\ (v_{11} - v_{22})^T \text{ (img 2)} \\ \vdots \end{bmatrix}}_{2n \times 6} \underbrace{b}_{6 \times 1} = 0
$$

**We solve this linear system with SVD**

**Vb = 0** where V is the 2n × 6 matrix above.

- Minimum 3 images needed (6 equations for 6 unknowns)
- Compute **SVD: V = UΣWᵀ**
- Solution: **b = last column of W** (corresponding to smallest singular value)
	- This refers to the principle component where data varies the least

Once you have b = [B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃]ᵀ, closed-form extraction:
$$
v_0 = (B_{12}B_{13} - B_{11}B_{23})/(B_{11}B_{22} - B_{12}^2)
$$$$
\lambda = B_{33} - [B_{13}^2 + v_0(B_{12}B_{13} - B_{11}B_{23})]/B_{11}
$$ $$
\alpha = \sqrt{\lambda/B_{11}}
$$$$
\beta = \sqrt{\lambda B_{11}/(B_{11}B_{22} - B_{12}^2)}
$$$$
\gamma = -B_{12}\alpha^2\beta/\lambda
$$$$
u_0 = \gamma v_0/\beta - B_{13}\alpha^2/\lambda
$$

Now you have **K**! We can also **extract extrinsics**

For each image i with homography Hᵢ = [h₁ h₂ h₃]:

$$
r_1 = \lambda K^{-1}h_1, \quad r_2 = \lambda K^{-1}h_2, \quad t = \lambda K^{-1}h_3
$$

where λ = 1/||K⁻¹h₁|| (for normalization)

$$
r_3 = r_1 \times r_2
$$

Then **$\mathbf{C}_{sw}$ = [r₁ r₂ r₃]** and **$\mathbf{t}_sw$ = t**
This lets us recover our full extrinsic matrix and recover from our assumption of $z_{w}=0$

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