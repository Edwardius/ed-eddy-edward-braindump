#stateEstimation
An iterative element to match two 3d objects on top of each other (for robotics, we usually use ICP in the terms of matching 3D pointclouds together).

# How it works
Mainly consists of two parts:
1. **Correspondence** for each point in the 3d pointcloud, match it to the closest point in the cloud you are matching to.
2. **Transformation** compute a transform that best aligns the corresponding points together.

# The Math
Given a source point cloud $\mathbf{P} = \{\mathbf{p}_i\}_{i=1}^{N}$ and a target point cloud $\mathbf{Q} = \{\mathbf{q}_j\}_{j=1}^{M}$, ICP finds the rotation matrix $\mathbf{R}$ and translation vector $\mathbf{t}$ that minimize: 
$$
E(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^{N} \|\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_{c(i)}\|^2
$$ where $c(i)$ denotes the index of the closest point in $\mathbf{Q}$ to the transformed point $\mathbf{R}\mathbf{p}_i + \mathbf{t}$.

## Correspondence
It will associate every point with the closest point in the target pointcloud
$$
c(i) = \arg\min_{j} \|\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_j\|
$$
## Transformation
There is both a closed and iterative form of the transformation step.

Given two pointclouds:
1. Correspond each point to a target point
2. Centralize the points about the respective centroid of each cloud
3. Compute the required rotation with SVD
4. Derive translation from the rotation

Given corresponding point pairs $\{(\mathbf{p}_i, \mathbf{q}_{c(i)})\}$, we want to minimize:

$$E(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^{N} \|\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_{c(i)}\|^2$$

1. Center the point clouds

First, compute the centroids:

$$\bar{\mathbf{p}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{q}_{c(i)}$$

Center the points:

$$\mathbf{p}_i' = \mathbf{p}_i - \bar{\mathbf{p}}, \quad \mathbf{q}_i' = \mathbf{q}_{c(i)} - \bar{\mathbf{q}}$$

2. Find the rotation using SVD

Compute the cross-covariance matrix (this is a $3\times 3$ matrix):

$$\mathbf{H} = \sum_{i=1}^{N} \mathbf{p}_i' (\mathbf{q}_i')^T$$

Perform SVD on $\mathbf{H}$ (this is a $3\times 3$ matrix, so computation is really fast):

$$\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

The optimal rotation is:

$$\mathbf{R} = \mathbf{V} \mathbf{U}^T$$

However, we need to ensure $\det(\mathbf{R}) = 1$ (proper rotation, not reflection):

$$\mathbf{R} = \mathbf{V} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(\mathbf{V}\mathbf{U}^T) \end{bmatrix} \mathbf{U}^T$$

3. Find the translation

Once we have $\mathbf{R}$, the translation is simply:

$$\mathbf{t} = \bar{\mathbf{q}} - \mathbf{R}\bar{\mathbf{p}}$$

**This is done iteratively until we reach some threshold error to stop at.** The SVD approach here is a closed-form approach for finding a transformation based on correspondence. However, its not final. We have to perform the transform / compute new correspondences, and compute a new transform.

#worldModeling