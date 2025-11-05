A form of [[Iterative Closest Point (ICP)]] that models each point as a gaussian distribution based on its local neighborhood.

# How it works
Instead of computing the distance error between each point, we calculate the extension of a [[Mahalanobis Distance]] between two a local distributions in the source and target.

# Distance between two distributions
People colloquially call this computation a [[Mahalanobis Distance]] even though its not the formal definition. More of an extension.

 Given two points, that we correspond to be two local distributions.
 $$
\mathbf{p}_{i} \sim \mathcal{N}(\boldsymbol{\mu}_{i},\boldsymbol{\Sigma}_{i})
$$
$$
\mathbf{q}_{j}\sim\mathcal{N}(\boldsymbol{\mu}_{j},\boldsymbol{\Sigma}_{j})
$$
Then the extended mahalanobis distance between them is

$$
D^{2}_{M}(\boldsymbol{\mu}_{i},\boldsymbol{\mu}_{j})=(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{j})^{T}(\boldsymbol{\Sigma}_{i}+\boldsymbol{\Sigma}_{j})^{-1}(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{j})
$$

> [!info] This no longer computes the number of standard deviations two points are from each other.

Instead, its better to think of this as a likelihood the two points are the same.
# Probabilistic Interpretation of GICP 
Say we have two local distributions. One is in the source cloud, another is in a target cloud.
$$
p_{i}=\mu_{p_{i}}+\epsilon_{p_{i}}\sim\mathcal{N}(\mu_{p_{i}},\Sigma_{p_{i}})
$$
$$
q_{i}=\mu_{q_{i}}+\epsilon_{q_{i}}\sim\mathcal{N}(\mu_{q_{i}},\Sigma_{q_{i}})
$$
These distributions are found by a k cluster around the points.

We then model the difference between the two points as.
$$
\Delta=p_{i}-q_{i}
$$
### Assumption: They are the same point
If we assume them to share the same true point, then we would witness the difference to have a mean of 0 and variance equivalent to the sum of the variances of the two distributions.
$$
p_{i}=x^{*}+\epsilon_{p_{i}} \;, \; \epsilon_{p_{i}}\sim\mathcal{N}(0,\Sigma_{p_{i}})
$$
$$
q_{i}=x^{*}+\epsilon_{q_{i}} \;, \; \epsilon_{q_{i}}\sim\mathcal{N}(0,\Sigma_{q_{i}})
$$
where $x^{*}$ is the true value of the point.
$$
\Delta_{i}=0+\epsilon_{p_{i}}-\epsilon_{q_{i}} \sim N(0,\Sigma_{p_{i}}+\Sigma_{q_{i}})
$$
This tells us that if the two points share the same true point, then their difference can be modeled by a Gaussian distribution with a $\mu=0$ and $\Sigma=\Sigma_{p_{i}}+\Sigma_{q_{i}}$.

Which means that we **know that we've successfully transformed the source cloud to the target when the likelihood of $N(0,\Sigma_{p_{i}}+\Sigma_{q_{i}})$ is at its max**
### What we are trying to find
Because the goal of GICP is to find a transform $T$ such that
$$
Tp_{i}=q_{i}
$$
$$
Tp_{i}=R\mu_{p_{i}}+t+\epsilon_{p_{i}}\sim\mathcal{N}(R\mu_{p_{i}}+t,R\Sigma_{p_{i}}R^{T})
$$
**We must find a transform on the source cloud that maximizes our [[Likelihood]] that the observed differences can be modeled by $N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}})$**
$$
T=\text{argmax}_{T}\mathcal{L}(T|\Delta)=\text{argmax}_{T}\prod_{i=1}^{n}\mathcal{L}(N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}) | \Delta_{i})
$$
Where
$$
\mathcal{L}(N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}) | \Delta_{i})\propto p(\Delta_{i}|N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}))
$$
$$
p(\Delta_{i}|N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}))=\frac{1}{(2\pi)^{3/2}(|R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}|)^{1/2}}\exp\left( -\frac{1}{2}\Delta_{i}^{T}(R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}})^{-1}\Delta_{i} \right)
$$
$$
\text{where}\;\;\Delta_i = R\mu_{p_i} + t - \mu_{q_i}
$$
Which is just the observation when put inside a gaussian distribution with mean 0 (since we are calculating how likely we can model the observation (with the transform) as part of a distribution assuming that the source and target share the same point.)

Because $\mathcal{L}(T|\Delta)$ is very expensive to calculate as a product, we can derive the **log-likelihood** to make it a summation, which is alot easier to optimize.

$$
T=\text{argmax}_{T}\log\mathcal{L}(T|\Delta)=\text{argmax}_{T}\log\prod_{i=1}^{n}\mathcal{L}(N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}) | \Delta_{i})
$$
$$
\log\mathcal{L}(T|\Delta)=\sum_{i=1}^{n}\log\mathcal{L}(N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}) | \Delta_{i})
$$
Taking the log of the probability...
$$
log\;p(\Delta_{i}|N(0,R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}))
$$
$$
=\log\left( \frac{1}{(2\pi)^{3/2}(|R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}}|)^{1/2}}\exp\left( -\frac{1}{2}\Delta_{i}^{T}(R\Sigma_{p_{i}}R^{T}+\Sigma_{q_{i}})^{-1}\Delta_{i} \right) \right)
$$
$$
-\log \mathcal{L}(T) = \sum_{i=1}^n \left[ \frac{3}{2}\log(2\pi) + \frac{1}{2}\log|R\Sigma_{p_i}R^T + \Sigma_{q_i}| + \frac{1}{2}\Delta_i^T(R\Sigma_{p_i}R^T + \Sigma_{q_i})^{-1}\Delta_i \right]
$$
We can drop the constants because they don't depend on T, **and GICP often drops the determinant term $\frac{1}{2}\log|R\Sigma_{p_i}R^T + \Sigma_{q_i}|$ as well**
$$
-\log \mathcal{L}(T) = \sum_{i=1}^n \left[\frac{1}{2}d_i^2(T) \right]\;,\;\text{where}\;\;d_{i}^{2}(T)=\Delta_i^T(R\Sigma_{p_i}R^T + \Sigma_{q_i})^{-1}\Delta_i
$$
**Which is where we finally end up getting the mahalanobis-like distance calculation for GICP**.

>[!info] all this to tell us that we are actually calculating the likelihood that the two points are the same true point. 
# How do we actually get this transformation
We want to solve
$$
T^* = \arg\min_T \sum_{i=1}^n d_i^2(T)
$$
$$
d_{i}^{2}(T)=\Delta_i^T(R\Sigma_{p_i}R^T + \Sigma_{q_i})^{-1}\Delta_i
$$
This is a **non-linear least squares problem**, which can be solved by a class of iterative optimizers.
There are many ways to solve for the transformation here. One way is [[Gauss-Newton Method]]

There are other methods, but I can't be bothered to go any further into this for now. small_gicp is fast not because of its optimizer, but rather because of its parallel processing architecture and a bunch of implementation optimizations.
