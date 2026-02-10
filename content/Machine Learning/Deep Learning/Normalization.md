# BatchNorm
Normalizes data to a $\mu=0 \;\;\sigma=1$ across a batch.
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
$$
y_i = \gamma \hat{x}_i + \beta
$$
$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i \quad \text{(batch mean)}
$$
$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2 \quad \text{(batch variance)}
$$
# LayerNorm
Normalizes across all features in each sample independently.
$$
\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}
$$
$$
y = \gamma \hat{x} + \beta
$$
$$
\mu_L = \frac{1}{H}\sum_{i=1}^{H} x_i \quad \text{(mean over features)}
$$
$$
\sigma_L^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu_L)^2 \quad \text{(variance over features)}
$$
