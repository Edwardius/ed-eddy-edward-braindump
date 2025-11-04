Given parameters, what is the probability of the observed data?
- **fixed parameters, variable observed data**
- $\mathcal{P}(\text{data}|\text{parameters})$
- Sums / integrates to 1

This differs from [[Likelihood]] as likelihood looks at a problem from the angle of fixed observed data, and finding the parameters that best explain the data.

# Example
I flipped a coin 10 times.
Given that a coin is fair ($p=0.5$), what is the probability of getting 7 heads?
$$
\mathcal{P}(7H|p=0.5)=\begin{pmatrix}10 \\
7\end{pmatrix}(0.5)^{10}=0.117
$$
What if we got 10 heads, whats the probability of that?
$$
\mathcal{P}(10H|p=0.5)=\begin{pmatrix}10 \\
10\end{pmatrix}(0.5)^{10}=0.00097
$$
