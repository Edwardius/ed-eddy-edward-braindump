Let $P(s) / G[z]$ be a real, rational, and proper transfer function.

Then the **impulse response** of $P(s) / G[z]$ is
$$
p(t)=\mathcal{L}^{-1}(P(s))
$$
$$
g[k]=\mathcal{Z}^{-1}(G[z])
$$
Some more nomenclature. 
$$
Y(s)=P(s)U(s) \iff y(t)=(p*u)(t)=\int ^{t}_{0}p(t-\tau)u(\tau)d\tau
$$
$$
Y[z]=G[z]U[z] \iff y[k]=(g*u)[k]=\sum ^{k}_{n=0}g[k-n]u[n]
$$
>[!info] Laplace and Z Transforms are linear. (distributive properly type shit)

