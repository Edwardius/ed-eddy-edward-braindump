Its a form of matrix decomposition that decomposes a $m * n$ matrix into three distinct matrices. 

$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$
$$
\mathbf{A} = \begin{bmatrix}
| & | & & | \\
\mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_m \\
| & | & & |
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & & & \\
& \sigma_2 & & \\
& & \ddots & \\
& & & \sigma_n
\end{bmatrix}
\begin{bmatrix}
— & \mathbf{v}_1^T & — \\
— & \mathbf{v}_2^T & — \\
& \vdots & \\
— & \mathbf{v}_n^T & —
\end{bmatrix}
$$
where $U$ is a $m*m$ matrix (left singular values), $V$ is a $n*n$ matrix (right singular values), $\Sigma$ is a diagonal matrix of eigenvalues in descending order.

# Low-Rank Approximation
The best rank-$k$ approximation of a matrix $A$ is given by keeping only the first $k$ largest singular values.
$$
\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T
$$
$$
\min_{\text{rank}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$
$$
\min_{\text{rank}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|_2 = \|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}
$$
This means that **no rank-k matrix can approximate matrix A better than a k-truncated SVD matrix**

# Applications
- compression
	- can also compress neural network weights!
- noise reduction
- dimensionality reduction

# How SVD is decomposed
1. Compute $A^{T}A$
$$
\mathbf{M} = \mathbf{A}^T\mathbf{A} \quad \text{(n × n symmetric positive semidefinite)}
$$
2. Find eigendecomposition of $A^TA$
$$
\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T
$$
3. Extract singular values
$$
\sigma_i = \sqrt{\lambda_i}
$$
4. Compute left singular values
$$
\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i \quad \text{for } \sigma_i > 0
$$
5. Assemble the decomposition
$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$
## Example: Manual Calculation

Let's compute SVD for a simple 2×2 matrix:
$$
\mathbf{A} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
$$
**Step 1**: Compute AᵀA
$$
\mathbf{A}^T\mathbf{A} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 10 & 6 \\ 6 & 10 \end{bmatrix}
$$
**Step 2**: Find eigenvalues
$$
\det(\mathbf{A}^T\mathbf{A} - \lambda\mathbf{I}) = \det\begin{bmatrix} 10-\lambda & 6 \\ 6 & 10-\lambda \end{bmatrix} = 0
$$
$$
(10-\lambda)^2 - 36 = 0 \implies \lambda_1 = 16, \lambda_2 = 4
$$

**Step 3**: Singular values
$$
\sigma_1 = \sqrt{16} = 4, \quad \sigma_2 = \sqrt{4} = 2
$$

**Step 4**: Find eigenvectors of AᵀA for V
$$
\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

**Step 5**: Compute U
$$
\mathbf{u}_1 = \frac{1}{\sigma_1}\mathbf{A}\mathbf{v}_1 = \frac{1}{4}\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$
$$
\mathbf{u}_2 = \frac{1}{\sigma_2}\mathbf{A}\mathbf{v}_2 = \frac{1}{2}\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

**Result**:
$$
\mathbf{A} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

>[!info] In practice, there are algorithmic implementations in like Numpy and PyTorch that you can use. The way above is not the standard way of doing it because eigendecomposition on very large matrices is very expensive