Any rational transfer function can be written as
$$
P(s) = P(\infty) + \sum_{i=1}^{n} \sum_{j=1}^{n_i} \frac{c_{i,j}}{(s - p_i)^j}
$$
And in Discrete Space
$$
G[z]=G[\infty]+\sum ^{n}_{i=1}\sum ^{n_{i}}_{j=1} \frac{c_{i,j}}{(z-p_{i})^j}
$$
Where $P(\infty)$ can mean different things in different contexts. But can be a useful characteristic. $n_{i}$ is the multiplicity of the pole $p_{i}$ (ie. $(s-3)^3$ has a multiplicity of 3).

Found by:
$$
P(\infty) = \lim_{ s \to \infty } P(s)
$$
Or by just doing the partial fraction decomposition and maybe a constant comes out of it.

### What we can tell from $P(\infty)$
The transfer function is:
- strictly proper if $P(\infty)$ is a constant
- proper if ${P(\infty)}$ is 0
- improper if $P(\infty)$ is $\infty$

