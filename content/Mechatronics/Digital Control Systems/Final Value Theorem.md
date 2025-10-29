# Final Value Theorem

The theorem states...
Let $p(t) / g[k]$ be a signal with Laplace/z transform that is real, rational, and proper. Then
1. If all poles of $P(s)/G[z]$ lie in $\mathbb{C}^{-}/\mathbb{D}$
$$
\lim_{ t \to \infty } p(t)=0
$$
$$
\lim_{ k \to \infty }g[k]=0 
$$
2. If all pose of $P(s) / G[z]$ like in $\mathbb{C}^{-}/\mathbb{D}$ excepts **for exactly one pole at 0/1**
$$
\lim_{ t \to \infty } p(t)=\lim_{ s \to \infty } sP(s)
$$
$$
\lim_{ k \to \infty } g[k]=\lim_{ z \to 1 } (z-1)G[z]
$$
> [!info] FOR G[z], THE FINAL VALUE THEOREM IS WHEN $z$ GOES TO 1
# Proof Setup 
$$g[k] = \mathcal{Z}^{-1}(G[z]) \;\;\text{(Definition of impulse response)}$$
$$= \mathcal{Z}^{-1}\left(G[\infty] + \sum_{i=1}^{n} \sum_{j=1}^{n_i} \frac{c_{i,j}}{(z-p_i)^j}\right)$$
$$= G[\infty]\mathcal{Z}^{-1}(1) + \sum_{i=1}^{n} \sum_{j=1}^{n_i} c_{i,j}\mathcal{Z}^{-1}\left(\frac{1}{(z-p_i)^j}\right)$$
$$= G[\infty] \cdot \delta[k] + \sum_{i=1}^{n} \sum_{j=1}^{n_i} c_{i,j} \frac{n(n-1)\dots(n-k+1)}{k!} p_i^{k-j}$$
# Proof of Part 1
$$

\text{Given: All poles of } G[z] \text{ lie in } \mathbb{D}

$$
$$

\text{WTS: } \lim_{k \to \infty} g[k] = 0

$$
$$

G[z] \text{ is real, rational, and proper}

$$
$$

\implies g[k] = G[\infty]\delta[k] + \sum_{i=1}^{n} \sum_{j=1}^{n_i} c_{i,j} \binom{k-1}{j-1} p_i^{k-j}

$$
where $\delta$ is 1 when $k=0$ and 0 otherwise. so that term is 0
$$

\implies \lim_{k \to \infty} g[k] = \lim_{k \to \infty} \sum_{i=1}^{n} \sum_{j=1}^{n_i} c_{i,j} \binom{k-1}{j-1} p_i^{k-j} = 0

$$
$$
where \begin{pmatrix}n \\
k\end{pmatrix} =
\frac{n(n-1)\dots(n-k+1)}{k!}
$$
# Proof of Part 2
$$

% Part (b)

\text{Given: All poles of } G[z] \text{ lie in } \mathbb{D} \text{ except exactly one at } z=1

$$
$$

\text{WTS: } \lim_{k \to \infty} g[k] = \lim_{z \to 1} (z-1)G[z]

$$
Let's specifically single out the pole at 1 from the summation
$$

\implies G[z] = G[\infty] + \frac{c_n}{z-1} + \sum_{i=1}^{n-1} \sum_{j=1}^{n_i} \frac{c_{i,j}}{(z-p_i)^j}

$$
$$

\implies g[k] = G[\infty]\delta[k] + c_n + \sum_{i=1}^{n-1} \sum_{j=1}^{n_i} c_{i,j} \binom{k-1}{j-1} p_i^{k-j}

$$
$$

\implies \lim_{k \to \infty} g[k] = c_n

$$
$$

\lim_{z \to 1} (z-1)G[z] = c_n = \lim_{k \to \infty} g[k]

$$
$$

% Corollary

Y[z] = G[z] \cdot \frac{z}{z-1}

$$
$$

\lim_{k \to \infty} y[k] = \lim_{z \to 1} (z-1)Y[z] = \lim_{z \to 1} zG[z] = G[1]

$$

# Corollary
Let $G[z] / P(s)$ be a real, ration, proper, and stable transfer function. Let $u[k] / u(t)$ be a unit step input to  such transfer functions. Then we will see that
$$
\lim_{ k \to \infty } y[k]=G[1]
$$
$$
\lim_{ t \to \infty } y(t)=P(0)
$$