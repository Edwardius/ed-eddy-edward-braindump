#probability 
It is a correction between sample and the population when measuring their mean and variance.
$$
N-1
$$
## Bessel's Correction Proof
**Goal**: Show that the unbiased estimator of population variance $\sigma^2$ requires division by $(n-1)$.
### Setup
- Population with mean $\mu$ and variance $\sigma^2$
- Random sample: $X_1, X_2, ..., X_n$ (i.i.d.)
- Sample mean: $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$
### The Question
Which estimator is unbiased for $\sigma^2$?
**Biased estimator**: 
$$
S_n^2 = \frac{1}{n}\sum_{i=1}^{n}(X_i - \bar{X})^2
$$
**Unbiased estimator**: 
$$
S_{n-1}^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2
$$

### Proof
We need to show: $E[S_{n-1}^2] = \sigma^2$
**Step 1**: Expand the sum of squared deviations
$$
\sum_{i=1}^{n}(X_i - \bar{X})^2 = \sum_{i=1}^{n}[(X_i - \mu) - (\bar{X} - \mu)]^2
$$
**Step 2**: Expand the square
$$
= \sum_{i=1}^{n}[(X_i - \mu)^2 - 2(X_i - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^2]
$$

**Step 3**: Distribute the summation

$$
= \sum_{i=1}^{n}(X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^{n}(X_i - \mu) + n(\bar{X} - \mu)^2
$$

**Step 4**: Simplify the middle term
Note that: 
$$
\sum_{i=1}^{n}(X_i - \mu) = \sum_{i=1}^{n}X_i - n\mu = n\bar{X} - n\mu = n(\bar{X} - \mu)
$$


$$
= \sum_{i=1}^{n}(X_i - \mu)^2 - 2n(\bar{X} - \mu)^2 + n(\bar{X} - \mu)^2
$$

$$
= \sum_{i=1}^{n}(X_i - \mu)^2 - n(\bar{X} - \mu)^2
$$

**Step 5**: Take expectations

$$
E\left[\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = E\left[\sum_{i=1}^{n}(X_i - \mu)^2\right] - E[n(\bar{X} - \mu)^2]
$$

**Step 6**: Evaluate each term

For the first term: 
$$
E\left[\sum_{i=1}^{n}(X_i - \mu)^2\right] = \sum_{i=1}^{n}E[(X_i - \mu)^2] = n\sigma^2
$$

For the second term, recall that $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$: 
$$
E[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \frac{\sigma^2}{n}
$$

Therefore: 
$$
E[n(\bar{X} - \mu)^2] = n \cdot \frac{\sigma^2}{n} = \sigma^2
$$

**Step 7**: Combine results

$$
E\left[\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2
$$

**Step 8**: Solve for the unbiased estimator

$$
E\left[\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2\right] = \frac{(n-1)\sigma^2}{n-1} = \sigma^2
$$
$$
\boxed{S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2 \text{ is an unbiased estimator of } \sigma^2}
$$

The division by $(n-1)$ compensates for using the sample mean $\bar{X}$ instead of the true mean $\mu$, which introduces bias by making the deviations artificially smaller.
