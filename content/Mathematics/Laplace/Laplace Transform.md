We usually take the **Laplace transform** of this plant to work in the **frequency domain**.
$$
\mathcal{L}=\int_{0}^{\infty}x(t)e^{-st}dt
$$
The **Inverse Laplace Transform** is given by
$$
x(t) = \mathcal{L}^{-1}\{X(s)\} = \frac{1}{2\pi j} \int_{\sigma - j\infty}^{\sigma + j\infty} X(s)e^{st} \, ds
$$
Following that, we can take the Laplace Transform of the dynamical system.