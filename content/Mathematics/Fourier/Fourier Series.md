Precursor to the [[Fourier Transform]] actually.

It shows that any periodic function can be represented as a superposition of sine and cosine functions as **harmonic frequencies**.

# Mathematical Definition

Given a periodic function $f(t)$ which has a period of $T$
$$
f(t)=\frac{a_{0}}{2}+\sum ^{\infty}_{n=1}\left[ a_{n}\cos\left( \frac{2\pi nt}{T} \right) + b_{n}\sin\left( \frac{2\pi nt}{T} \right)\right]
$$
Where coefficients $a_{n}$ and $b_{n}$ are found by:
$$
a_n = \frac{2}{T} \int_0^T f(t) \cos\left(\frac{2\pi n t}{T}\right) dt
$$
$$
b_n = \frac{2}{T} \int_0^T f(t) \sin\left(\frac{2\pi n t}{T}\right) dt
$$

![[Pasted image 20251029143751.png]]

# Extension to Non-Periodic
Its pretty simple, you just:
1. pretend that the bounds of your non-periodic function is a single period
2. fit a Fourier Series on that pseudo-periodic function ($T=L$)
3. the series will match $f(t)$ perfectly within the bounds $[0,L]$ and just repeat forever after that.

