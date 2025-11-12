Given two PDFs over the same random variable, we might want to quantify how 'far' they are.

# Kullback-Leibler Divergence
$$
KH(p_{2}||p_{1})=-\int p_{2}(\mathbf{x})\ln\left( \frac{p_{1}(\mathbf{x})}{p_{2}(\mathbf{x})} \right)d\mathbf{x}\geq
0
$$
Only equals to 0 when $p_{1}=p_{2}$

This is not a very visual way of seeing things. Something for more visual understanding would be
# Quantile-Quantile Plot

Graphs the CDF of two probability functions against each other.
#probability 