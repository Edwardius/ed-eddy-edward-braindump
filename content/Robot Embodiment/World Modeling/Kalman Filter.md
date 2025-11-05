A recursive algorithm used to estimate the state of a **linear** system based on noisy measurements.
# Setup
There's a ton of variables to keep track of:
- $\mathbf{x}_{k} \in \mathbb{R}^n$ is the state vector at time $k$
- $\mathbf{F}_k \in \mathbb{R}^{n \times n}$ is the state transition matrix, how state changes overtime **on its own (irrespective of the input)**
- $\mathbf{B}_k \in \mathbb{R}^{n \times m}$ is the control input matrix, maps control input to its effect on the state
- $\mathbf{u}_k \in \mathbb{R}^m$ is the control input
- $\mathbf{z}_k \in \mathbb{R}^p$ is the measurement vector
- $\mathbf{H}_k \in \mathbb{R}^{p \times n}$ is the measurement matrix, maps the true state to what the sensor should measure (without noise)
- $\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$ is the process noise
- $\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R}_k)$ is the measurement noise

The goal of the Kalman Filter is to estimate a hidden state of the linear system from the noisy measurements that we get.
$$
\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k
$$
$$
\mathbf{z}_{k}=\mathbf{H}_{k}\mathbf{x}_{k}+\mathbf{v}_{k}
$$
> [!info] Hidden state refers to the "underlying" state of the system. It is the true state of the system. We never have a direct understanding of this state in real life, we only have noisey measurements that get us there.
# Two-steps
A Kalmann Filter has two main steps:
### Predict Step
First we predict the current hidden state, and its covariance from the previous state and its covariance.
$$
\hat{\mathbf{x}}_{k|k-1}=\mathbf{F}_{k}\hat{\mathbf{x}}_{k-1|k-1}+\mathbf{B}_{k}\mathbf{u}_{k}
$$

### Update Step