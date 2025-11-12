Given observed data, how well do different parameters explain it?
- **Fixed data, variable parameters**
- $\mathcal{L}(\text{parameters}|\text{data})$
- Does NOT sum to 1

This is different from probability because we are trying to fit parameters to observed data

# Example
I flipped a coin 10 times.
If I got heads 7 times, whats the likelihood that the probability of flipping heads was 0.5?
$$
\mathcal{L}(p=0.5|7H)=\begin{pmatrix}10  \\
7\end{pmatrix}(0.5)^{10}=0.117
$$
> [!info] This is where the two definitions overlap, when the fixed data and parameters are the same for each. BUT, whats different is what we are varying after that. For likelihood, we are varying the behavior of the coin itself.

What's the likelihood that the probability of flipping heads was 0.7?
$$
\mathcal{L}(p=0.7|7H)=\begin{pmatrix}10  \\
7\end{pmatrix}(0.7)^{10}=0.267
$$
**What does this tell us?**  That it's more likely that the probability of flipping heads was 0.7 not 0.5.
#probability 