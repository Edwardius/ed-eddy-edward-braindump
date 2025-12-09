This is a totally different method of supervised learning. Its not some addon to logistic regression.

![[Pasted image 20251209181136.png]]

It is mainly formulated for **Binary Classification**

Given training data
$$
\{ (\mathbf{x}_{1}, y_{1}),(\mathbf{x}_{2},y_{2}),\dots,(\mathbf{x}_{n}, y_{n}) \}
$$
where
$$
\mathbf{x}_{i}\in \mathbb{R}^{n}\;\;y_{i}\in \{ -1, +1 \}
$$
We are trying to solve the following optimization problem
$$
\underset{ \mathbf{w},b }{ min } \frac{1}{2}||\mathbf{w}||^{2} \;\;\text{subject to}\;\;y_{i}(\mathbf{w}^{T}\mathbf{x}_{i}+b)\geq 1
$$
> [!info] To work with multiple classes, do a one vs rest approach

#machineLearning