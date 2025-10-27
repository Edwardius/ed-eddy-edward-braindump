This document outlines some basic translations between matrix and component form to allow me to derive things properly.

# Basic Matrix-Vector Product
**Matrix Form**
$$
y=Wx
$$
**Explicit Representation**
$$
W = \begin{bmatrix} W_{11} & W_{12} & W_{13} \\ W_{21} & W_{22} & W_{23} \\ W_{31} & W_{32} & W_{33} \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
$$
**Computing Each Element**
$$
y_1 = W_{11}x_1 + W_{12}x_2 + W_{13}x_3
$$
$$
y_2 = W_{21}x_1 + W_{22}x_2 + W_{23}x_3 
$$
$$
y_3 = W_{31}x_1 + W_{32}x_2 + W_{33}x_3
$$
**Component Form**
$$
y_i = \sum_{j} W_{ij} x_j
$$
- The index j **gets summed out**
- The hanging index, i, becomes the index of the output

# Matrix-Matrix Product
**Matrix Form**
$$
C=AB
$$
**Explicit Representation**
$$
C = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\
A_{31} & A_{32} \end{bmatrix} \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{21} & B_{22}  & B_{23}\end{bmatrix}
$$
**Computing Each Element**
$$
C_{11}=A_{11}B_{11}+A_{12}B_{21}
$$
$$
C_{12}=A_{11}B_{12}+A_{12}B_{22}
$$
$$
C_{13}=A_{11}B_{13}+A_{12}B_{23}
$$
$$
C_{21}=A_{21}B_{11}+A_{22}B_{21}
$$
$$
C_{22}=A_{21}B_{12}+A_{22}B_{22}
$$
$$
C_{23}=A_{21}B_{13}+A_{22}B_{23}
$$
$$
C_{31}=A_{31}B_{11}+A_{32}B_{21}
$$
$$
C_{32}=A_{31}B_{12}+A_{32}B_{22}
$$
$$
C_{33}=A_{31}B_{13}+A_{32}B_{23}
$$
**Component Form**
$$
C_{i,j}=\sum_{k}A_{i,k}B_{k,j}
$$
- **k gets summed out!!**
# Inner and Outer Product
$$
\text{inner product } c = a^{T}b
$$
$$
\text{outer product }C=ab^T
$$
>[!info] In code, these both return a dot product (same value). You have to explicitly try to turn it into an outer product. `a.unsqueeze(1)@b.unsqueeze(0) or torch.outer(a,b)`

**Component Form**
$$
\text{inner product (dot product) }c=\sum _{i}a_{i}b_{i}
$$
$$
C_{ij}=a_{i}b_{j}
$$
# Transpose
**Matrix Form**
$$
B = A^T
$$
**Explicit**
$$
A = \begin{bmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \end{bmatrix}
$$
**Transpose**
$$
B = A^T = \begin{bmatrix} A_{11} & A_{21} \\ A_{12} & A_{22} \\ A_{13} & A_{23} \end{bmatrix}
$$
**Component Form**
$$
B_{i,j}=A_{j,i}
$$
- indexes get flipped

# Complex Translations
**Matrix Form**
$$
y=Wx+b
$$
**Explicit**
$$
\begin{pmatrix} y_{1} \\
y_{2} \\
\dots \\
y_{n} \end{pmatrix} = \begin{pmatrix}W_{11} & W_{12} & \dots  & W_{1m} \\
W_{21} & W_{22} & \dots  & W_{2m} \\
W_{31} & W_{32} & \dots  & W_{3m} \\
\dots \\
W_{n1} & W_{n2} & \dots  & W_{nm}
\end{pmatrix} \cdot\begin{pmatrix}x_{1} \\ x_{2}  \\
x_{3} \\ \dots \\ x_{m}\end{pmatrix}+ \begin{pmatrix}b_{1} \\
b_{2} \\
b_{3} \\
\dots \\
b_{n}\end{pmatrix}
$$
**Component Form**
$$
y_{i}=\sum W_{i,j}x_{j}+b_{i}
$$
- Notice how j is summed out (and so is x for that manner)
>[!info] When we are working with batches, we ideally want Y to be shape [b, n] given that X is of shape [b, m] coming in. As a result, we shuffle the linear layer equation a bit to make things smoother.

$$
Y=XW^T+b
$$
This is because $[b,n]=[b,m]@[m, n]+[n]$ and so less code to write.
$$
Y_{ij}=\sum_{k}X_{ik}W_{kj}+b_{j}
$$
Note that b, the bias is broadcasted. 

# Quadratic Form (Matrix and Vector)
**Matrix Form**
$$
z=x^TAx
$$
**Explicit**
$$
z=\begin{pmatrix}x_{1} \\ x_{2}  \\
x_{3} \\ \dots \\ x_{n}\end{pmatrix}^{T}

\begin{pmatrix}W_{11} & W_{12} & \dots  & W_{1n} \\
W_{21} & W_{22} & \dots  & W_{2n} \\
W_{31} & W_{32} & \dots  & W_{3n} \\
\dots \\
W_{n1} & W_{n2} & \dots  & W_{nn}
\end{pmatrix}

\begin{pmatrix}x_{1} \\ x_{2}  \\
x_{3} \\ \dots \\ x_{n}\end{pmatrix}
$$
**Component Form**
This collapses to a single value.
$$
z=\sum^n_{i}\sum ^n_{j}x_{i}A_{i,j}x_{j}
$$
# Quadratic Form (Matrix and Matrix)
**Matrix Form**
$$
Z=X^TAX
$$
**Explicit**
$$
\begin{pmatrix}Z_{11} & \dots  & Z_{1n} \\
Z_{21} & \dots  & Z_{2n} \\
\dots \\
Z_{n1} & \dots  & Z_{nn}
\end{pmatrix}
=
\begin{pmatrix}X_{11} & X_{12} & \dots  & X_{1m} \\
X_{21} & X_{22} & \dots  & X_{2m} \\
\dots \\
X_{n1} & X_{n2} & \dots  & X_{nm}
\end{pmatrix}^T

\begin{pmatrix}W_{11} & W_{12} & \dots  & W_{1m} \\
W_{21} & W_{22} & \dots  & W_{2m} \\
W_{31} & W_{32} & \dots  & W_{3m} \\
\dots \\
W_{m1} & W_{m2} & \dots  & W_{mm}
\end{pmatrix}

\begin{pmatrix}X_{11} & X_{12} & \dots  & X_{1m} \\
X_{21} & X_{22} & \dots  & X_{2m} \\
\dots \\
X_{n1} & X_{n2} & \dots  & X_{nm}
\end{pmatrix}
$$
**Component Form**
Its just the smaller form but its done on every element of Z
$$
Z_{pq}=\sum^n_{i}\sum ^n_{j}x_{i,p}A_{i,j}x_{j,q}
$$
- note how i and j get summed out.
# Trace
**Matrix Form**
AKA the SUM of the diagonal
$$
z = tr(A)
$$
**Component Form**
$$
z=\sum _{i}A_{ii}
$$
# Trace of Product
**Matrix Form**
$$
z=tr(AB)
$$
**Explicit**
$$
z=tr\left( \sum _{k}A_{i,k}B_{k, j} \right)
$$
Two things to note, A and B must both be the same dimensions (one needs to be the transpose of the other.). So A in NxM and B is MxN. I only care about
$$
z=tr\left( C_{ii}=\sum _{j}A_{ij}B_{ji} \right)
$$
I only care about the diagonal so...
**Component Form**
$$
z=\sum _{i} \sum_{j}A_{ij}B_{ji}
$$
# Element-Wise Operations
**Matrix Form**
$$
C = A ⊙ B
$$
**Component Form**
$$
C_{ij}=A_{ij}\cdot B_{ij}
$$
- A and B gotta be the same size in theoretical land, in coding, broadcasting can be done

# Summary

| Operation                    | Explicit Example                                      | Component Form                                      |
| ---------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| $\mathbf{y} = W\mathbf{x}$   | $y_1 = W_{11}x_1 + W_{12}x_2$                         | $y_i = \sum_j W_{ij}x_j$                            |
| $C = AB$                     | $C_{11} = A_{11}B_{11} + A_{12}B_{21}$                | $C_{ik} = \sum_j A_{ij}B_{jk}$                      |
| $B = A^T$                    | $B_{12} = A_{21}$                                     | $B_{ij} = A_{ji}$                                   |
| $C = \mathbf{ab}^T$          | $C_{12} = a_1 b_2$                                    | $C_{ij} = a_i b_j$                                  |
| $z = \mathbf{a}^T\mathbf{b}$ | $z = a_1b_1 + a_2b_2$                                 | $z = \sum_i a_i b_i$                                |
| $C = A \odot B$              | $C_{11} = A_{11}B_{11}$                               | $C_{ij} = A_{ij}B_{ij}$                             |
| $z=x^TAx$                    | $z=x_{1}A_{1,2}x_{2}+x_{2}A_{2,2}x_{2}$               | $z=\sum_{i}^n\sum_{j}^nx_{i}A_{i,j}x_{j}$           |
| $Z=X^TAX$                    | $z_{4,3}=x_{1,4}A_{1,2}x_{2,3}+x_{2,4}A_{2,2}x_{2,3}$ | $z_{p,q}=\sum_{i}^n\sum_{j}^nx_{i,p}A_{i,j}x_{j,q}$ |
