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
