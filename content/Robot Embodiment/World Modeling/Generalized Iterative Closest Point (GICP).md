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


