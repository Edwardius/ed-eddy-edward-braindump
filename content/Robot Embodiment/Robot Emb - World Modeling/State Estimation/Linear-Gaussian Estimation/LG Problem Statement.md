This folder consists of various methods to estimate *discrete-time, linear, time-varying* equations. This sort of problem is called a *Linear Gaussian (LG)* model.

They can be characterized as the following:
$$
\text{model:} \;\;\mathbf{x}_{k}=\mathbf{A}_{k-1}\mathbf{x}_{k-1}+\mathbf{v}_{k}+\mathbf{w}_{k}
$$
$$
\text{observation model:} \;\;\mathbf{y}_{k}=\mathbf{C}_{k}\mathbf{x}_{k}+\mathbf{n}_{k}
$$
Where $k$ is a index in discrete time.
- $\mathbf{x}_{k} \in \mathbb{R}^{N}$ is the state of the system 
- $\mathbf{x}_{0} \in \mathbb{R}^{N} \sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{\hat{P}}_{0})$ is the initial state of the system 
- $\mathbf{v}_{k} \in \mathbb{R}^{N}$ input to the system. might have a mapping to $\mathbf{v}_{k}=\mathbf{B}\mathbf{u}_{k}$
- $\mathbf{w}_{k} \in \mathbb{R}^{N}\sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{Q}_{k})$ process noise
- $\mathbf{y}_{k} \in \mathbb{R}^{N}$ measurement
- $\mathbf{n}_{k} \in \mathbb{R}^{N}\sim\mathcal{N}(\mathbf{\hat{x}}_{0}, \mathbf{R}_{k})$ measurement noise

- $\mathbf{A}_{k}$ is the state transition matrix
- $\mathbf{C}_{k}$ is the observation matirx which maps our state to our measurement

**The problem for state estimation is as follows:**
*The problem of state estimation is to come up with an estimate $\hat{\mathbf{x}_{k}}$ of the true state of a system, at one or more timesteps, $k$, given knowledge of the initial state, $\mathbf{x}_{0}$, a sequence of measurements $y_{0:K}$ a sequence of inputs $\mathbf{v}_{1:K}$ as well as knowledge of the system's motion and observation models*

There are roughly two paradigms to solving this:
- [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Bayesian Inference]] prior, posterior paradigm where we are updating a prior density (based on our initial state, inputs, and motion model) with our measurements to produce a posterior estimate
	- [[Generalized Gaussian Filter]] [[Bayes Filter]] [[Kalman Filter]]
- [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Maximum A Posteriori]] here we are using optimization to find the most likely posterior state given the information we have.

This problem can be solved with many approaches, one of them being the [[Kalman Filter]]

#linearGaussianEstimation
