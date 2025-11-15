Given a gaussian estimate of our entire path (from anything solving [[NLNG Problem Statement]] or [[LG Problem Statement]])
$$
\mathcal{N}(\hat{\mathbf{x}}_{k},\hat{P}_{k}) \text{ for } k=1\dots K
$$
And a ground truth, $x_{true,k}$ we can compute a straightforward error
$$
\hat{e}_{k}=\hat{x}_{k}-x_{true,k}
$$
![[Pasted image 20251114210915.png]]

A estimate is considered **unbiased** if $\hat{e}_{k}$ hovers around zero and **consistent** if the error stays within $3\sigma_{k}$ where $\sigma_{k}=\sqrt{ \hat{P}_{k} }$ 99.7% of the time.

>[!error] We usually measure error over a single trial throughout all of its timesteps, instead of having multiple trials. This is because we assume that the average of the error overtime is the same as the average of the error over many trials (known as **ergodic hypothesis**)

# Unbias
We require that
$$
E[\hat{e}_{k}]=0
$$

# Consistent
We require that
$$
E\left[ \frac{\hat{e}_{k}^{2}}{\hat{P}_{k}} \right]=1
$$
