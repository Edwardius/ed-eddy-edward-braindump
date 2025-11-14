>[!error] This is not the canonical way to solve a batch of states. It's just a middle man.

This is one way, doing a sparse Cholesky decomposition followed by forward and backward passes.

A **Cholesky Decomposition** decomposes  a matrix into a lower-triangular representation and its transpose

$$ 
\mathbf{A} = \begin{bmatrix} 4 & 12 & -16 \\ 12 & 37 & -43 \\ -16 & -43 & 98 \end{bmatrix} = \begin{bmatrix} 2 & 0 & 0 \\ 6 & 1 & 0 \\ -8 & 5 & 3 \end{bmatrix} \begin{bmatrix} 2 & 6 & -8 \\ 0 & 1 & 5 \\ 0 & 0 & 3 \end{bmatrix} 
$$
To do a Cholesky Decomposition, its pretty easy, a pattern forms that allows you to compute it in linear time.

![[Pasted image 20251113195604.png]]