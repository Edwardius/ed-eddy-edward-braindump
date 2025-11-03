A self-supervised learning technique that tries to learn representations of the data where similar inputs are encoded closer together in the latent space than highly dissimilar data.

# Contrastive Loss Functions
## Objective
We want to learn an encoding such that
$$
\text{sim}(f(x), f(x^+)) > \text{sim}(f(x), f(x^-))
$$
where $sim()$ is some arbitrary similarity function like cosine similarity.
## InfoNCE Loss
The more similar things are, the closer we get to 1, but negative log causes it to go to 0.
$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}
$$