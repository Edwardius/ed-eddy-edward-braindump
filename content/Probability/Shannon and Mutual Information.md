Once we have modeled a [[Basic Probability Nomenclature|PDF]] of some random variable, we might want to quantify how certain we are with the PDF's parameters.

Two ways

# Shannon Information
$$
H(\mathbf{x})=-E[\ln p(\mathbf{x})]=-\int p(\mathbf{x})\ln p(\mathbf{x})d\mathbf{x}
$$
# Mutual Information
$$
I(\mathbf{x},\mathbf{y})=E\left[ \ln\left( \frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})} \right) \right]=\int \int p(\mathbf{x},\mathbf{y})\ln\left(\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}\right)d\mathbf{x}d\mathbf{y}
$$
When $\mathbf{x}$ and $\mathbf{y}$ are independent, then
$$
I(\mathbf{x},\mathbf{y})
=\int \int p(\mathbf{x},\mathbf{y})\ln\left(\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}\right)d\mathbf{x}d\mathbf{y}
=\int \int p(\mathbf{x},\mathbf{y})\underbrace{ \ln\left(\frac{p(\mathbf{x})p(\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}\right) }_{ 0 }d\mathbf{x}d\mathbf{y}
=0
$$
Any bit of dependence makes it greater than 0.

# Together
$$
I(\mathbf{x},\mathbf{y})=H(\mathbf{x})+H(\mathbf{y})-H(\mathbf{x},\mathbf{y})
$$

#probability 