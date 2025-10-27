##### ReLU - Rectified Linear Unit
$$
\mathrm{Re}LU(z)=max(0, z)
$$
Its gradient is
$$\frac{\partial \mathrm{Re}LU}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \ \\
 0 & \text{if } z \leq 0 \end{cases}$$
 Remember! This is a vector! So it becomes a vector of 1's and 0's

##### Leaky ReLU - Leaky Rectified Linear Unit
$$
\mathrm{Re}LU(z)=max(\alpha z, z)
$$
Its gradient is
$$\frac{\partial\mathrm{Re}LU}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \ \\
 \alpha & \text{if } z \leq 0 \end{cases}$$
 Remember! This is a vector! So it becomes a vector of 1's and  $\alpha$'s
##### Sigmoid
$$
\sigma (z)=\frac{1}{1+e^{ -z }}
$$
Its gradient is
$$
\frac{\partial\sigma}{\partial z}=\sigma(z)(1-\sigma(z))
$$
Remember! This is a vector! So we are performing this operation on each of the elements.
##### Softmax
Usually used on the output layer.
$$
softmax(z)_{i}=\frac{e^{z_{i} }}{\sum ^{C}_{j=1}e^{ z_{j} }}
$$
Its **Jacobian** is
$$\frac{\partial softmax(z)_i}{\partial z_j} = \begin{cases} softmax(z)_i(1 - softmax(z)_i) & \text{if } i = j \\
 -softmax(z)_i softmax(z)_j & \text{if } i \neq j \end{cases}$$
Thing is, if we pair Softmax with cross entropy loss, this gradient simplifies to:
$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$
Which is why you end up seeing that `nn.CrossEntropyLoss` has a builtin softmax.

We have:

- $\hat{y} = \text{softmax}(z)$, so $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- Loss: $\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$
- Goal: find $\frac{\partial \mathcal{L}}{\partial z_k}$ where $k$ denotes an arbitrary index of z (we need to find all $z_{k}$ to form $\frac{\partial \mathcal{L}}{\partial z}$ )

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

**Chain rule to get gradient w.r.t. pre-softmax**

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{i=1}^{C} \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_k}$$

**Substitute what we know**

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{i=1}^{C} \left(-\frac{y_i}{\hat{y}_i}\right) \cdot \frac{\partial \hat{y}_i}{\partial z_k}$$

**Split the sum into two cases**

The Jacobian of softmax is: $$\frac{\partial \hat{y}_i}{\partial z_k} = \begin{cases} \hat{y}_i(1 - \hat{y}_i) & \text{if } i = k \\ -\hat{y}_i \hat{y}_k & \text{if } i \neq k \end{cases}$$
> [!info] this is because we have $\hat{y} = \text{softmax}(z)$!!!! All the activation functions correlate like this, its how the whole thing connects together.
$$
a_{i}=\sigma (z)=\frac{1}{1+e^{ -z }}
$$
>[!info] so
$$
\frac{\partial a_{i}}{\partial z}=\frac{\partial\sigma}{\partial z}=\sigma(z)(1-\sigma(z))=a_{i}(1-a_{i})
$$
>[!info] Anyways, back to it

So:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \left(-\frac{y_k}{\hat{y}_k}\right) \cdot \hat{y}_k(1 - \hat{y}_k) + \sum_{i \neq k} \left(-\frac{y_i}{\hat{y}_i}\right) \cdot (-\hat{y}_i \hat{y}_k)$$

First term: $$-\frac{y_k}{\hat{y}_k} \cdot \hat{y}_k(1 - \hat{y}_k) = -y_k(1 - \hat{y}_k) = -y_k + y_k \hat{y}_k$$

Second term: $$\sum_{i \neq k} \left(-\frac{y_i}{\hat{y}_i}\right) \cdot (-\hat{y}_i \hat{y}_k) = \sum_{i \neq k} y_i \hat{y}_k = \hat{y}_k \sum_{i \neq k} y_i$$

**Combine**

$$\frac{\partial \mathcal{L}}{\partial z_k} = -y_k + y_k \hat{y}_k + \hat{y}_k \sum_{i \neq k} y_i$$

$$= -y_k + \hat{y}_k\left(y_k + \sum_{i \neq k} y_i\right)$$
$y_{k}$ here is just inserted into the summation and...
$$= -y_k + \hat{y}_k \sum_{i=1}^{C} y_i$$
Since $y$ is one-hot encoded: $\sum_{i=1}^{C} y_i = 1$

$$\frac{\partial \mathcal{L}}{\partial z_k} = -y_k + \hat{y}_k \cdot 1 = \hat{y}_k - y_k$$
**In vector form:**
$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

