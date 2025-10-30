# CT Plant Systems (Generalized)
The dynamics of a typical plant in continuous time can be represented as
$$
\sum_{i=0}^{n} a_i \frac{d^i y}{dt^i}(t) - \sum_{j=0}^{m} b_j \frac{d^j u}{dt^j}(t) = 0
$$
$y$ and $u$ here are arbitrary output/input of the plant. $a$ and $b$ are arbitrary coefficients.

We usually take the **Laplace transform** of this plant to work in the **frequency domain**.
$$
\mathcal{L}=\int_{0}^{\infty}x(t)e^{-st}dt
$$
The **Inverse Laplace Transform** is given by
$$
x(t) = \mathcal{L}^{-1}\{X(s)\} = \frac{1}{2\pi j} \int_{\sigma - j\infty}^{\sigma + j\infty} X(s)e^{st} \, ds
$$
Following that, we can take the Laplace Transform of the dynamical system.
$$
\mathcal{L} \implies\sum_{i=0}^{n} a_i s^i Y(s) - \sum_{j=0}^{m} b_j s^j U(s) = 0
$$
And from this obtain the transfer function.
$$
Y(s) = \frac{\sum_{j=0}^{m} b_j s^j}{\sum_{i=0}^{n} a_i s^i} U(s)
$$
$$
P(s):=\frac{Y(s)}{U(s)} = \frac{\sum_{j=0}^{m} b_j s^j}{\sum_{i=0}^{n} a_i s^i}
$$
> [!info] $:=$ is not the assignment operator like you see in [[Optimizers]] and gradient descent, it just mean "is defined as" in formal math terms.

So from this we see that we have derived a nicely defined characteristic equation of the dynamical system in the frequency domain.
### Example
Mass spring damper (as seen in [[Control Systems]]) would be represented as.
$$
m\ddot{y} + c\dot{y} + ky = F
$$
Which in terms of the general formula would look something like.
$$
a_2\ddot{y} + a_1\dot{y} + a_0y - b_0u = 0
$$
where $a_{2}=m$, $a_{1}=c$, $a_{0}=k$, and $F=u \; \text{(THIS IS OUR INPUT)}),b_{0}=1$. 
We can take a Laplace transform of above to then determine the transfer function of the system.

# DT Plant Systems (Generalized)
The dynamics of a typical plant in discrete time can be represented as
$$
\sum_{i=0}^{n} a_i y[k+i] - \sum_{j=0}^{m} b_j u[k+j] = 0
$$
Where again $y$ and $u$ are the output/input of the system respectively. $a$ and $b$ are arbitrary coefficients characterizing the plant.

Just like how in CT we took the Laplace Transform to bring ourselves into the frequency domain to do better processing, we can do a similar thing here. In Discrete-Time, we are doing a **Z transform**.
$$
X[z]=\mathcal{Z}(x[k])=\sum ^{\infty}_{k=0}x[k] \frac{1}{z^{k}}
$$
The **Inverse Z Transform** is given by
$$
x[k] = \mathcal{Z}^{-1}(X[z]) = \frac{1}{2\pi j} \oint_{\text{unit circle}} X[z]z^{k-1} \, dz = \frac{1}{2\pi} \int_{-\pi}^{\pi} X[e^{j\theta}]e^{jk\theta} \, d\theta
$$
Following that, we can take the **Z transform** of the platnt.
$$
\mathcal{Z}\implies \sum_{i=0}^{n} a_i z^i Y[z] - \sum_{j=0}^{m} b_j z^j U[z] = 0
$$
And just like in the frequency domain, we can rearrange to get a discrete transfer function.
$$
Y[z] = \frac{\sum_{j=0}^{m} b_j z^j}{\sum_{i=0}^{n} a_i z^i} U[z]
$$
$$
G[z]:=\frac{Y[z]}{U[z]} = \frac{\sum_{j=0}^{m} b_j z^j}{\sum_{i=0}^{n} a_i z^i} 
$$
# Parallels

| Continuous-Time             | Discrete-Time            |
| --------------------------- | ------------------------ |
| Differential equation (ODE) | Difference equation      |
| Laplace transform           | z-transform              |
| Transfer function $P(s)$    | Transfer function $G[z]$ |
| $Y(s) = P(s)U(s)$           | $Y[z] = G[z]U[z]$        |
