##### MSE - Mean Squared Error
$$
\mathcal{L}=\frac{1}{2}\sum_{i=0}^{C}(\hat{y}_{i} - y_{i})^{2}=\frac{1}{2}\sum_{i=0}^{C}(\hat{y}^{2}_{i}-2\hat{y}_{i}y_{i}+y^{2}_{i})
$$
$$
\frac{\partial\mathcal{L}}{\partial \hat{y}}=\frac{1}{2}(2\hat{y}-2y)=\mathbf{\hat{y}-y}
$$
So it becomes just the difference between the prediction and the target!

##### BCE - Binary Cross Entropy
Assuming y can either be 0 or 1.
$$
\mathcal{L} = -\sum_{i=0}^{C}[y_{i}\log \hat{y}_{i}+(1-y_{i})\log{(1-\hat{y}_{i})}]
$$
$$
\frac{\partial\mathcal{L}}{\partial \hat{y}}=\frac{\hat{y}-y}{\hat{y}(1-\hat{y})}
$$
##### Cross Entropy Loss
This is assuming $y$ is a one-hot vector, it is the generalized version of BCE.
$$
\mathcal{L}=-\sum ^{C}_{i=1} y_{i}\log \hat{y}_{i}
$$
$$
\frac{\partial\mathcal{L}}{\partial \hat{y}}=-\frac{y}{\hat{y}}
$$
> Notation is pretty important here. Need to stop thinking in scalars and instead in vectors. 
