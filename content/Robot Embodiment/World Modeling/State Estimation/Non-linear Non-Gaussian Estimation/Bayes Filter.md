see [[NLNG Problem Statement]]

The Bayes filter seeks to come up with an entire [[Basic Probability Nomenclature|PDF]] to represent the likelihood of state $\mathbf{x}_{k}$, using only the measurements up to and including the current time. This is shown as the notation
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}})
$$
Because [[NLNG Problem Statement]] is [[Markovian]], we have independence between measurements, so
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}})=\eta p(\mathbf{y}_{k}|\mathbf{x}_{k})p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k-1}})
$$
If we integrate over all possible values of $\mathbf{x}_{k-1}$ we get
$$
p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k-1}})=\int p(\mathbf{x}_{k}, \mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},y_{0:k-1})d\mathbf{x}_{k-1}
$$
$$
=\int p(\mathbf{x}_{k}| \mathbf{x}_{k-1},\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},y_{0:k-1})p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},y_{0:k-1})d\mathbf{x}_{k-1}
$$
Taking advantage of [[Markovian]] property again.
$$
p(\mathbf{x}_{k}| \mathbf{x}_{k-1},\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},y_{0:k-1})=p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k})
$$
$$
p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},y_{0:k-1})d\mathbf{x}_{k-1}=p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1})
$$
Which leads us to the **Bayes Filter**
$$
\underbrace{ p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}}) }_{ \text{posterior belief} }=\eta \underbrace{ p(\mathbf{y}_{k}|\mathbf{x}_{k}) }_{ \substack{\text{observation} \\ \text{correction} \\ \text{using}\;\mathbf{g}(\cdot)} }\int \underbrace{ p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) }_{ \substack{\text{motion prediction} \\ \text{using }\mathbf{f}(\cdot)} }\underbrace{ p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1}) }_{ \text{prior belief} }d\mathbf{x}_{k-1}
$$