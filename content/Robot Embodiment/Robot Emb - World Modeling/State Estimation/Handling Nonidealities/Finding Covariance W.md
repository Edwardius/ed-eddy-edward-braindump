From [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] we have the loss function:
$$
J(\mathbf{x})=\frac{1}{2}\sum ^{N}_{i=1}\mathbf{e}_{i}(\mathbf{x})^{T}\mathbf{W}^{-1}_{i}(\mathbf{e}_{i}(\mathbf{x}))
$$
But how do we set $\mathbf{W}_{i}$?  

Previously we were assuming that $\mathbf{W}_{v,k}$ and $\mathbf{W}_{y,k}$ can be thought of as positive-definite symmetric matrix weights **that are often set to the process noise and measurement noise covariances of the system**

So
$$
\mathbf{W}_{v,k}=\mathbf{Q}_{k}\quad \mathbf{W}_{y,k}=\mathbf{R}_{k}
$$
$$
\mathbf{W} = \text{diag}(\mathbf{W}_v, \mathbf{W}_y), \quad \mathbf{W}_v = \text{diag}(\mathbf{W}_{v,0}, \ldots, \mathbf{W}_{v,K})
$$
$$
\mathbf{W}_y = \text{diag}(\mathbf{W}_{y,0}, \ldots, \mathbf{W}_{y,K})
$$

But this assumes that we know the process noise and measurement noise beforehand.

>[!caution] We could use a datasheet, but that often isn't reliable

# Supervised Covariance Estimation
Given we have a set of $K$ groundtruth state values, $\mathbf{x}_{true}$. We can compute the process noise and measurement noise as:
$$
\mathbf{Q}=\frac{1}{K-1}\sum_{k=1}^{K}(\mathbf{e}_{v,k}(\mathbf{x}_{true})-\bar{\mathbf{e}}_{v})(\mathbf{e}_{v,k}(\mathbf{x}_{true})-\bar{\mathbf{e}}_{v})^{T}
$$
$$
\mathbf{R}=\frac{1}{K-1}\sum_{k=1}^{K}(\mathbf{e}_{y,k}(\mathbf{x}_{true})-\bar{\mathbf{e}}_{v})(\mathbf{e}_{y,k}(\mathbf{x}_{true})-\bar{\mathbf{e}}_{v})^{T}
$$
$$
\bar{\mathbf{e}}_{v}=\frac{1}{K}\sum_{k=1}^{K} \mathbf{e}_{v,k}(\mathbf{x}_{true}) \quad \bar{\mathbf{e}}_{y}=\frac{1}{K+1}\sum_{k=1}^{K} \mathbf{e}_{y,k}(\mathbf{x}_{true})
$$
where from [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]]
$$
\mathbf{e}_{v,k}(\mathbf{x})=\begin{cases}
\check{\mathbf{x}}_{0}-\mathbf{x}_{0} & k=0 \\
\mathbf{f}(\mathbf{x}_{k-1},\mathbf{v}_{k},\mathbf{0}) & k=1\dots K
\end{cases}
$$$$
\mathbf{e}_{y,k}(\mathbf{x})=\mathbf{y}_{k}-\mathbf{g}(\mathbf{x}_{k},\mathbf{0})\;\; k=0\dots K
$$

Once we have characterized the noise, we can proceed to using them in a real operational scenario.

# Adaptive Covariance Estimation
Using a trailing window of $L$ datapoints from the current time, $k$, we can adaptively change our covariances according to

**Measurement Noise Covariance**
$$
\bar{\mathbf{e}}_{y,k} = \frac{1}{L} \sum_{\ell=k-1}^{k-L} \mathbf{e}_{y,\ell}
$$
$$
\mathbf{S}_{y,k} = \frac{1}{L-1} \sum_{\ell=k-1}^{k-L} (\mathbf{e}_{y,\ell} - \bar{\mathbf{e}}_{y,k})(\mathbf{e}_{y,\ell} - \bar{\mathbf{e}}_{y,k})^T
$$
$$
\mathbf{R}_k = \mathbf{S}_{y,k} - \frac{1}{L} \sum_{\ell=k-1}^{k-L} \mathbf{G}_\ell \mathbf{P}_\ell \mathbf{G}_\ell^T
$$
**Process Noise Covariance**
$$
\bar{\mathbf{e}}_{v,k} = \frac{1}{L} \sum_{\ell=k-1}^{k-L} \mathbf{e}_{v,\ell}
$$
$$
\mathbf{S}_{v,k} = \frac{1}{L-1} \sum_{\ell=k-1}^{k-L} (\mathbf{e}_{v,\ell} - \bar{\mathbf{e}}_{v,k})(\mathbf{e}_{v,\ell} - \bar{\mathbf{e}}_{v,k})^T
$$
$$
\mathbf{Q}_k = \mathbf{S}_{v,k} - \frac{1}{L} \sum_{\ell=k-1}^{k-L} \mathbf{F}_{\ell-1} \mathbf{P}_{\ell-1} \mathbf{F}_{\ell-1}^T
$$
