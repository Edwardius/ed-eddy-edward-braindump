Extension of [[Extended Kalman Filter]] where in the update step, we just keep correcting until we reach some threshold.

# Update ONLY
Compute the first order approximation of the non-linear measurement function using our predicted state prior.
$$
\mathbf{H}_{k}=\frac{ \partial h }{ \partial \mathbf{x} } \bigg|_{\hat{\mathbf{x}}_{k|k-1}}
$$
Compute the Kalman Gain
$$
\mathbf{K}_{k}=\frac{\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}}{\mathbf{H}_{k}\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}+\mathbf{R}_{k}}
$$
Update the priors with the measurement using the Kalman Gain (this becomes the posterior)
$$
\hat{\mathbf{x}}_{k|k}=\hat{\mathbf{x}}_{k|k-1}+\mathbf{K}_{k}(\mathbf{z}_{k}-\underbrace{ h(\hat{\mathbf{x}}_{k|k-1}) }_{ \substack{\text{We use} \\ \text{non-linear} \\ \text{here} }})
$$
$$
\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{H}_{k}\mathbf{K}_{k})\mathbf{P}_{k|k-1}
$$
$$
\mathbf{x}_{op,k}\leftarrow \hat{\mathbf{x}}_{k|k}
$$
second iteration

Compute the first order approximation of the non-linear measurement function using our predicted state prior.
$$
\mathbf{H}_{k}=\frac{ \partial h }{ \partial \mathbf{x} } \bigg|_{\mathbf{x}_{op,k}}
$$
Compute the Kalman Gain
$$
\mathbf{K}_{k}=\frac{\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}}{\mathbf{H}_{k}\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}+\mathbf{R}_{k}}
$$
Update the priors with the measurement using the Kalman Gain (this becomes the posterior)
$$
\hat{\mathbf{x}}_{k|k}=\mathbf{x}_{op,k}+\mathbf{K}_{k}(\mathbf{z}_{k}-\underbrace{ h(\mathbf{x}_{op,k}) }_{ \substack{\text{We use} \\ \text{non-linear} \\ \text{here} }})
$$
$$
\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{H}_{k}\mathbf{K}_{k})\mathbf{P}_{k|k-1}
$$
$$
\mathbf{x}_{op,k}\leftarrow \hat{\mathbf{x}}_{k|k}
$$
repeat...
keep doing this until $\Delta \mathbf{x}_{op,k}<threshold$
