#probability #bayesTheorem #setTheory 

# Probability Density Function
Let $p(x)$ be a random variable $x$. The probability density function can be defined as.
$$
\int ^{b}_{a}p(x)dx
$$
Which tells us
$$
p(a\leq x\leq b)
$$
The whole distribution adds to 1
$$
\int ^{\infty}_{-\infty}p(x)dx=1
$$
### [[00 - Multivariate Calculus Table of Contents|Multivariate Representation]]
This is if we have a probability density function in multiple dimensions.
$$
\int ^{\mathbf{b}}_{\mathbf{a}}p(\mathbf{x})d\mathbf{x}=\int \int \int \int p(x_{1},x_{2},x_{3},..,x_{N})dx_{1}dx_{2}dx_{3}\dots dx_{N}
$$
# Cumulative Density Function
$$
\int ^{b}_{-\infty}p(x)dx=p(x\leq b)
$$

# Joint probability
$$
p(\mathbf{x},\mathbf{y})=p(\mathbf{x}|\mathbf{y})p(\mathbf{y})=p(\mathbf{y}|\mathbf{x})p(\mathbf{x})
$$
This gives you [[Bayes' Theorem]]

