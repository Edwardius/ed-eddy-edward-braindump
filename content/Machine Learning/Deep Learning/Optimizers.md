## **The General Form**
$$
\theta_{\text{new}} = \theta_{\text{old}} + \Delta\theta
$$
where $\Delta\theta$ is computed from the gradient $\frac{\partial L}{\partial \theta}$.

# Stochastic Gradient Descent (SGD)
$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \frac{\partial L}{\partial \theta}
$$
where:
- $\eta$ = learning rate (hyperparameter, e.g., 0.001)
- $\frac{\partial L}{\partial \theta}$ = gradient you computed
- negative because gradient is calculated to be in the direction of greater loss
# SGD with Momentum
Adds "velocity" to smooth updates:
$$
v_t = \beta v_{t-1} + \frac{\partial L}{\partial W}
$$
$$
W_{\text{new}} = W_{\text{old}} - \eta v_t
$$
where:
- $v_t$ = velocity (exponential moving average of gradients)
- $\beta$ = momentum coefficient (e.g., 0.9)

# Adam
Combines momentum + adaptive learning rates:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \frac{\partial L}{\partial W}
$$ 
(first moment, like momentum)
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \left(\frac{\partial L}{\partial W}\right)^2
$$ 
(second moment, squared gradients)

**Bias correction:** 
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
**Update:** 
$$
W_{\text{new}} = W_{\text{old}} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
where:
- $\beta_1 = 0.9$ (first moment decay)
- $\beta_2 = 0.999$ (second moment decay)
- $\epsilon = 10^{-8}$ (numerical stability)
- **Adam adapts learning rate per parameter based on gradient history.**
