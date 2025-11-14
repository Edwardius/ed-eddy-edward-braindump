[[Bayes Filter]] Provides us a fundamental framework for designing frameworks. However, it is very high-level, and its generalized to all PDF. We don't need something so generalized in Engineering ;) so we can look specifically at a subset of filters that **assume Gaussian [[Basic Probability Nomenclature|PDFs]]** up front.

Recall Bayes Filter is
$$
\underbrace{ p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}}) }_{ \text{posterior belief} }=

\eta \underbrace{ p(\mathbf{y}_{k}|\mathbf{x}_{k}) }_{ \substack{\text{observation} \\ \text{correction} \\ \text{using}\;\mathbf{g}(\cdot)} }

\int \underbrace{ p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) }_{ \substack{\text{motion prediction} \\ \text{using }\mathbf{f}(\cdot)} }

\underbrace{ p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1}) }_{ \text{prior belief} }d\mathbf{x}_{k-1}
$$
In general, we begin by assuming a Gaussian prior at time $k-1$
$$
p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1})=\mathcal{N}(\hat{\mathbf{x}}_{k-1},\hat{\mathbf{P}}_{k-1})
$$
We the assume that passing this though a non-linear motion model $\mathbf{f}(\cdot)$ is gonna give us another Gaussian
$$
p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k})=\mathcal{N}(\check{\mathbf{x}}_{k},\check{\mathbf{P}}_{k})
$$
