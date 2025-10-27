You have computed the loss from a loss function and now you want to propagate that loss (error) up through the network to improve the next round of inference. That's what **backpropagation** does.

>[!info] To be completely honest, this would make things so much cleaner if I knew matrix calculus... [[The Matrix Cookbook]] ... actually nevermind, I just went down this rabbit hole and it doesn't fully let me understand Deep Learning. DL has some elementwise operations that fuck things up.

^^ Above is not going to help my problems as much as know [[Matrix and Component Forms]]
# Chain Rule, lots of it
Backpropagation is just one extremely long chain rule. 

> [!warn] This example is just the basic representation of backpropagation that most courses teach. I went out of my way to really understand how we actually do this in [[Multivariate Chain Rule]] which I wrote as a generalized understanding of deriving multivariate gradients. This then led into a generalized method to deriving gradients with a "Derive as you go" mentality, which is what people used as the basis of Computational Graphs like [[PyTorch Computational Graph]]. I suggest myself to read this section briefly, its got alot of holes, and I see it now because its very hard to teach DL without an understanding of how the gradients are derived. (I think alot of people brush over the derivation of gradients because its way too complex)

Given a function:
$$
f(g(x))
$$
$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$
# Walk-through
Say we have a model consisting of linear layers and sigmoid activations in between them. Note we are doing one set of $x$, $y$ , and $\hat{y}$. $m$ is the number of input features. $C$ is the number of classes.
$$
x=\begin{pmatrix}x_{0} \\
x_{1} \\
x_{2} \\
x_{3} \\
\dots \\
x_{m}\end{pmatrix},

\hat{y}=\begin{pmatrix}\hat{y}_{0} \\
\hat{y}_{1} \\
\dots \\
\hat{y}_{C}\end{pmatrix},

y=\begin{pmatrix}y_{0} \\
y_{1} \\
\dots \\
y_{C}\end{pmatrix}
$$
If this were a real optimization, then we might accumulate the loss and gradients of a batch and propagate in one go. $z$ is the **output vector of a linear layer**, $W$ is the **weights matrix** of the linear layer, $b$ is the **bias vector** of the linear layer, $\sigma$ is an **arbitrary activation function** (commonly denotes sigmoid, but lets pretend). These are all linear algebra operations.
$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$
$$
a^{[1]}=\sigma(z^{[1]})
$$
$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$
$$
\hat{y}=\sigma(z^{[2]})
$$
$$
\mathcal{L}=loss(\hat{y},y)
$$
**Loss here is a single scalar value, as we analyze all the prediction and target attributes together**
### The first gradient we know is
$$
\frac{\partial\mathcal{L}}{\partial \hat{y}}=\begin{pmatrix} \frac{\partial\mathcal{L}}{\partial \hat{y}_{0}}  \\
\frac{\partial\mathcal{L}}{\partial \hat{y}_{1}} \\
\dots \\
\frac{\partial\mathcal{L}}{\partial \hat{y}_{C}}\end{pmatrix}
$$
Which is the partial derivative of the loss function with respect to the output prediction (target is a constant, duh). **This is a gradient!!!! A VECTOR OF PARTIAL DERIVATIVES** It will compute a vector of partial derivatives with respect to the value of $\hat{y}_{i}$.

#### Some common loss functions and their partial derivatives
more in [[Loss Functions]], here's some for now.
### Going Backwards
Now that we **literally know a physical value for the gradient** $\frac{\partial\mathcal{L}}{\partial \hat{y}}$. We can work backwards.
### Constructing the Chain Rule Throughout
So since we know $\frac{\partial\mathcal{L}}{\partial \hat{y}}$ as a concrete vector of numbers, we can send this gradient back through the network. The "trunk" of the backpropagation is traveling through the I/O, branches of the backpropagation are going into the parameters and biases (which is what we want).
$$
z^{[1]}=W^{[1]}x+b^{[1]}
$$
$$
a^{[1]}=\sigma(z^{[1]})
$$
$$
z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$
$$
\hat{y}=\sigma(z^{[2]})
$$
$$
\mathcal{L}=loss(\hat{y},y)
$$
 To make the network learn, we want to update:
$$
\frac{\partial\mathcal{L}}{\partial W^{[1]}}, \frac{\partial\mathcal{L}}{\partial b^{[1]}} ,\frac{\partial\mathcal{L}}{\partial W^{[2]}} ,\frac{\partial\mathcal{L}}{\partial b^{[2]}}
$$
We don't know what these gradients are, but we do know the loss gradient. **We can propagate the loss gradient up the network through the variables that are passed from one layer to another ($a$ and $z$)**
$$
\frac{\partial\hat{y}}{\partial z^{[2]}}= \frac{\partial \hat{y}}{\partial\sigma}\cdot \frac{\partial \sigma}{\partial z^{[2]}}=\frac{\partial \sigma}{\partial z^{[2]}}=\hat{y}(1-\hat{y})
$$
$$
\frac{\partial\mathcal{L}}{\partial z^{[2]}}=\frac{\partial\mathcal{L}}{\partial \hat{y}}\cdot \frac{\partial\hat{y}}{\partial z^{[2]}}
$$
$$
\frac{\partial\mathcal{L}}{\partial a^{[1]}}=\frac{\partial\mathcal{L}}{\partial z^{[2]}}\cdot \frac{\partial z^{[2]}}{\partial a^{[1]}}
$$
$$
\frac{\partial\mathcal{L}}{\partial z^{[1]}}= \frac{\partial\mathcal{L}}{\partial a^{[1]}}\cdot \frac{\partial a^{[1]}}{\partial z^{[1]}}
$$
$$
\frac{\partial a^{[1]}}{\partial z^{[1]}}=\frac{\partial a^{[1]}}{\partial\sigma}\cdot \frac{\partial\sigma}{\partial a^{[1]}}=\frac{\partial\sigma}{\partial a^{[1]}}=z^{[1]}(1-z^{[1]})
$$
$$
\frac{\partial \mathcal{L}}{\partial x}= \frac{\partial\mathcal{L}}{\partial z^{[1]}}\cdot \frac{\partial z^{[1]}}{\partial x}
$$
All the weight and bias gradients can be computed from this.

**Better to evaluate as you go.** What you can do is
1. Compute the loss gradient
2. Backward through activation
3. Compute weight gradient
4. Compute bias gradient
5. Pass gradient to next layer
6. repeat 2-5

This chain fundamentally inspired the existence of a [[PyTorch Computational Graph]].
### Some common activation functions and their partial derivatives
When it comes to the inner parts of the network, every **partial derivative from here on out is a jacobian**. However, some of the activation functions are element-wise operations, which become a **diagonal jacobian matrix**.

more in [[Activation Functions]], here are some for now
### Some Layers and their Partial Derivatives
more in [[Layers]], here's a handful of them

## After gradients are computed, we Optimize!
we then use our [[Optimizers]] to update the weights by the gradients we found.