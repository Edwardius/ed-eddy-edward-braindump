The Schur complement is defined for a **block matrix** $M$ that is partitioned into four sub-blocks:
$$
M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}
$$
where $A$ and $D$ are square matrices.
### Schur Complement of $\mathbf{D}$
If the block $D$ is **invertible**, the **Schur complement of $D$ in $M$** is the matrix $\mathbf{M/D}$ defined as:
$$
\mathbf{M/D} = A - B D^{-1} C
$$
### Schur Complement of $\mathbf{A}$
Similarly, if the block $A$ is **invertible**, the **Schur complement of $A$ in $M$** is the matrix $\mathbf{M/A}$ defined as:
$$
\mathbf{M/A} = D - C A^{-1} B
$$

# Decomposition
The cool thing is that if the complement is possible, then we can decompose $\mathbf{M}$

## Decomposition via Schur Complement of $\mathbf{A}$

This decomposition is possible if the top-left block $\mathbf{A}$ is **invertible**.
The **Schur Complement of $A$ in $M$** is:
$$
\mathbf{M/A} = D - C A^{-1} B
$$
The matrix $M$ can then be decomposed as:
$$
M = \begin{bmatrix} I & 0 \\ C A^{-1} & I \end{bmatrix} \begin{bmatrix} A & 0 \\ 0 & \mathbf{M/A} \end{bmatrix} \begin{bmatrix} I & A^{-1} B \\ 0 & I \end{bmatrix}
$$

Where:
- The first matrix is **block lower triangular** ($L$).
- The second matrix is **block diagonal** ($D$) and contains the original block $A$ and the Schur complement $\mathbf{M/A}$.
- The third matrix is **block upper triangular** ($U$).

This formula shows that $M$ is **congruent** to the block-diagonal matrix $\mathrm{diag}(A, \mathbf{M/A})$ via a block triangular matrix.

## Decomposition via Schur Complement of $\mathbf{D}$

This decomposition is possible if the bottom-right block $\mathbf{D}$ is **invertible**.
The **Schur Complement of $D$ in $M$** is:
$$
\mathbf{M/D} = A - B D^{-1} C
$$
The matrix $M$ can then be decomposed as:
$$
M = \begin{bmatrix} I & B D^{-1} \\ 0 & I \end{bmatrix} \begin{bmatrix} \mathbf{M/D} & 0 \\ 0 & D \end{bmatrix} \begin{bmatrix} I & 0 \\ D^{-1} C & I \end{bmatrix}
$$

Where:
- The first matrix is **block upper triangular** ($U'$).
- The second matrix is **block diagonal** ($D'$) and contains the Schur complement $\mathbf{M/D}$ and the original block $D$.
- The third matrix is **block lower triangular** ($L'$).