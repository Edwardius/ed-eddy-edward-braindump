Extension of [[Robot Embodiment/World Modeling/State Estimation/Linear-Gaussian Estimation/Bayesian Inference|Bayesian Inference]] except for [[NLNG Problem Statement]]. 

See [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] perspective of inference as well. We will be using [[Gauss-Newton Method]]

Following our model, we can linearize it about an operating point.
$$
\mathbf{x}_k \approx \mathbf{f}(\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, \mathbf{0}) + \mathbf{F}_{k-1}(\mathbf{x}_{k-1} - \mathbf{x}_{\text{op},k-1}) + \mathbf{w}_k'
$$
$$
\mathbf{F}_{k-1} = \frac{\partial \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{v}_k, \mathbf{w}_k)}{\partial \mathbf{x}_{k-1}}\bigg|_{\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, 0}, \quad \mathbf{G}_k = \frac{\partial \mathbf{g}(\mathbf{x}_k, \mathbf{n}_k)}{\partial \mathbf{x}_k}\bigg|_{\mathbf{x}_{\text{op},k}, 0}
$$
We can lift this into lifted matrix form like in the previous [[Robot Embodiment/World Modeling/State Estimation/Linear-Gaussian Estimation/Bayesian Inference|Bayesian Inference]]

