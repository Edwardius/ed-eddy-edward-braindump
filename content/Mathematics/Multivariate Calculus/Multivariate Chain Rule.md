# Multivariate Chain Rule
Given a composition of functions:
$$
\mathcal{L}=\mathcal{L}(f(\theta))
$$
Where:
- $\theta$ is your **parameter** (can be scalar, vector, matrix, tensor - any shape)
- $f(\theta)$ is some **intermediate function**(can output scalar, vector, matrix, tensor - any shape)
- $\mathcal{L}$ is the **final output** (can be scalar, vector, matrix, tensor - any shape)

**The universal multivariable chain rule is given as:**
$$
\frac{\partial\mathcal{L}}{\partial\theta_{indicies}} = \sum_{\text{all indicies of f}} \frac{\partial\mathcal{L}}{\partial f_{\text{all indicies of f}}} \cdot \frac{\partial f_{\text{all indicies of f}}}{\partial\theta_{\text{indicies}}}
$$
**To translate:** In order to determine how $\mathcal{L}$ changes with an element $\theta_{i}$, we need to sum up the contributions of all the elements of $f$  that depend on $\theta_{i}$. 
## Why "Sum over all indices of $f$"?
Because $f$ is the intermediate quantity that connects $\theta$ to $L$:
- Each element $f_{\beta}$ depends on $\theta$
- $L$ depends on each element $f_{\beta}$
- To find how $L$ depends on $\theta$, you add up contributions from all the $f_{\beta}$
## Examples
### **Both are vectors**
- $\theta = \mathbf{x} \in \mathbb{R}^n$ (parameter vector)
- $f(\mathbf{x}) = \mathbf{y} \in \mathbb{R}^m$ (intermediate vector)
- $L(\mathbf{y}) \in \mathbb{R}$ (final scalar)
$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$
Sum over all $m$ components of the intermediate quantity $\mathbf{y}$.

### **Parameter is matrix, intermediate is vector**
- $\theta = W \in \mathbb{R}^{p \times q}$ (parameter matrix)
- $f(W) = \mathbf{y} \in \mathbb{R}^m$ (intermediate vector)
- $L(\mathbf{y}) \in \mathbb{R}$ (final scalar)
$$\frac{\partial L}{\partial W_{ab}} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{ab}}$$
Sum over all $m$ components of the intermediate quantity $\mathbf{y}$.

### **Parameter is matrix, intermediate is matrix**
- $\theta = W \in \mathbb{R}^{p \times q}$ (parameter matrix)
- $f(W) = Y \in \mathbb{R}^{m \times k}$ (intermediate matrix)
- $L(Y) \in \mathbb{R}$ (final scalar)
$$\frac{\partial L}{\partial W_{ab}} = \sum_{i=1}^{m} \sum_{j=1}^{k} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial W_{ab}}$$
Sum over all $m \times k$ components of the intermediate quantity $Y$.

## How does chain rule work here?
Suppose we have: 
$$L = L(f_3(f_2(f_1(\theta))))$$
Where:
- $\theta$ is your parameter
- $f_1(\theta)$ is the first intermediate quantity
- $f_2(f_1(\theta))$ is the second intermediate quantity
- $f_3(f_2(f_1(\theta)))$ is the third intermediate quantity
- $L$ is the final scalar

We apply chain rule repeatedly, but as separate summations

**Step 1**: From $f_3$ to $f_2$ 
$$\frac{\partial L}{\partial (f_2)_{\beta}} = \sum_{\gamma} \frac{\partial L}{\partial (f_3)_{\gamma}} \cdot \frac{\partial (f_3)_{\gamma}}{\partial (f_2)_{\beta}}$$
**Step 2**: From $f_2$ to $f_1$ 
$$\frac{\partial L}{\partial (f_1)_{\alpha}} = \sum_{\beta} \frac{\partial L}{\partial (f_2)_{\beta}} \cdot \frac{\partial (f_2)_{\beta}}{\partial (f_1)_{\alpha}}$$
**Step 3**: From $f_1$ to $\theta$ 
$$\frac{\partial L}{\partial \theta_{\text{indices}}} = \sum_{\alpha} \frac{\partial L}{\partial (f_1)_{\alpha}} \cdot \frac{\partial (f_1)_{\alpha}}{\partial \theta_{\text{indices}}}$$
If you substitute Step 1 into Step 2 into Step 3:

$$\frac{\partial L}{\partial \theta_{\text{indices}}} = \sum_{\alpha} \sum_{\beta} \sum_{\gamma} \frac{\partial L}{\partial (f_3)_{\gamma}} \cdot \frac{\partial (f_3)_{\gamma}}{\partial (f_2)_{\beta}} \cdot \frac{\partial (f_2)_{\beta}}{\partial (f_1)_{\alpha}} \cdot \frac{\partial (f_1)_{\alpha}}{\partial \theta_{\text{indices}}}$$
Of course, deriving like this is a nightmare, which is why we need some sort of easier way to do it. For deep learning, we can follow a **derive as you go** pattern. In computing, this is usually done with some graph structure like the [[PyTorch Computational Graph]].

Once you get to a certain point of writing out the component representation of the gradients with the given functions, that's when you hit a wall of "how do I actually simplify this?". See [[The Art of Simplifying Multivariate Gradients]]