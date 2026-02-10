In an attempt to mitigate [[Spectral Bias]], this method maps input x coordinates into a set of random Fourier Features.

# Stemming from Fourier Series
We start with a [[Fourier Series]]

$$
f(x) = \sum_{n=-\infty}^{\infty} \left[a_n \cos\left(\frac{2\pi n x}{L}\right) + b_n \sin\left(\frac{2\pi n x}{L}\right)\right]
$$

Generalize to multiple dimensions. Given $x \in\mathbb{R}^{d}$

$$
f(\mathbf{x}) = \sum_{n=-\infty}^{\infty} \left[a_n \cos\left(\frac{2\pi n \cdot \mathbf{x}}{L}\right) + b_n \sin\left(\frac{2\pi n \cdot \mathbf{x}}{L}\right)\right]
$$

Replace harmonic frequencies with arbitrary frequencies (**because we don't know what these frequencies are beforehand**). Instead of $n=\dots-1,0, 1, 2, \dots$ we use arbitrary frequency vectors $\omega_{i} \in \mathbb{R}^{d}$

$$
f(\mathbf{x}) = \sum_{i=1}^{m} \left[a_i \cos(2\pi \mathbf{\omega}_i \cdot \mathbf{x}) + b_i \sin(2\pi \mathbf{\omega}_i \cdot \mathbf{x})\right]
$$

Because we don't know these harmonic frequencies, we **randomly sample** them. The current best choice to do so is sampling using a **Gaussian Distribution**.

$$
\mathbf{\omega}_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
$$
$$
f(\mathbf{x}) \approx \sum_{i=1}^{m} \left[a_i \cos(2\pi \mathbf{\omega}_i \cdot \mathbf{x}) + b_i \sin(2\pi \mathbf{\omega}_i \cdot \mathbf{x})\right]
$$
Meaning that we are making the assumption that we can represent our output as approximately a sum of sine and cosine functions at differing magnitudes at a set of enough randomly sampled frequencies.

Given that we have a bunch of $\omega$, we can finally stack our component form into a matrix representation of our mapping.
$$
\gamma(\mathbf{x}) = \begin{bmatrix} \cos(2\pi \mathbf{\omega}_1 \cdot \mathbf{x}) \\ \sin(2\pi \mathbf{\omega}_1 \cdot \mathbf{x}) \\ \cos(2\pi \mathbf{\omega}_2 \cdot \mathbf{x}) \\ \sin(2\pi \mathbf{\omega}_2 \cdot \mathbf{x}) \\ \vdots \\ \cos(2\pi \mathbf{\omega}_m \cdot \mathbf{x}) \\ \sin(2\pi \mathbf{\omega}_m \cdot \mathbf{x}) \end{bmatrix} \in \mathbb{R}^{2m}
$$
And we are trying to find
$$
f(x)=w^{T}\gamma(x), \;\; \text{where} \; w=\begin{pmatrix}a_{1} & b_{1}  & a_{2} &  b_{2} & \dots & a_{m} & b_{m}\end{pmatrix}
$$
Which in a nice way looks like a Linear Layer! (see [[Layers]])

Where we're essentially learning the coefficients of a Fourier Series sampled at random frequencies $\omega$