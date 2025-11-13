Can be derived from [[Exploiting Sparsity in Batch Solution]] with the Cholesky Decomposition.

Given the [[LG Problem Statement]] we can batch solve the problem recursively by **RTS** smoother.

**Forward**: this is a [[Kalman Filter]]
$(k = 1\dots K)$
Predict:
$$
\check{\mathbf{P}}_{k,f} = \mathbf{A}_{k-1}\hat{\mathbf{P}}_{k-1}\mathbf{A}_{k-1}^T + \mathbf{Q}_k
$$
$$
\check{\mathbf{x}}_{k,f} = \mathbf{A}_{k-1}\hat{\mathbf{x}}_{k-1} + \mathbf{v}_k
$$
$$
\mathbf{K}_k = \check{\mathbf{P}}_{k,f}\mathbf{C}_k^T(\mathbf{C}_k\check{\mathbf{P}}_{k,f}\mathbf{C}_k^T + \mathbf{R}_k)^{-1}
$$
Update:
$$
\hat{\mathbf{P}}_{k,f} = (\mathbf{I} - \mathbf{K}_k\mathbf{C}_k)\check{\mathbf{P}}_{k,f}
$$
$$
\hat{\mathbf{x}}_{k,f} = \check{\mathbf{x}}_{k,f} + \mathbf{K}_k(\mathbf{y}_k - \mathbf{C}_k\check{\mathbf{x}}_{k,f})
$$

**Backward**:
$(k = K \ldots 1)$
$$
\hat{\mathbf{x}}_{k-1} = \hat{\mathbf{x}}_{k-1,f} + \left(\hat{\mathbf{P}}_{k-1,f}\mathbf{A}_{k-1}^T\check{\mathbf{P}}_{k,f}^{-1}\right)(\hat{\mathbf{x}}_k - \check{\mathbf{x}}_{k,f})
$$
$$
\hat{\mathbf{P}}_{k-1} = \hat{\mathbf{P}}_{k-1,f} + \left(\hat{\mathbf{P}}_{k-1,f}\mathbf{A}_{k-1}^T\check{\mathbf{P}}_{k,f}^{-1}\right)\left(\hat{\mathbf{P}}_k - \check{\mathbf{P}}_{k,f}\right) \times \left(\hat{\mathbf{P}}_{k-1,f}\mathbf{A}_{k-1}^T\check{\mathbf{P}}_{k,f}^{-1}\right)^T
$$

which are initialized with
$$
\hat{\mathbf{P}}_{0,f} = (\mathbf{I} - \mathbf{K}_0\mathbf{C}_0)\check{\mathbf{P}}_0
$$
$$
\hat{\mathbf{x}}_{0,f} = \check{\mathbf{x}}_0 + \mathbf{K}_0(\mathbf{y}_0 - \mathbf{C}_0\check{\mathbf{x}}_0)
$$
$$
\hat{\mathbf{x}}_K = \hat{\mathbf{x}}_{K,f} 
$$
$$
\hat{\mathbf{P}}_K = \hat{\mathbf{P}}_{K,f}
$$

and $\mathbf{K}_0 = \check{\mathbf{P}}_0\mathbf{C}_0^T(\mathbf{C}_0\check{\mathbf{P}}_0\mathbf{C}_0^T + \mathbf{R}_0)^{-1}$

>[!error] This is the **canonical method** to solving a batch of states at once, without any approximation involved
>

>[!error] Historically, RTS Smoother was after the [[Kalman Filter]] was introduced