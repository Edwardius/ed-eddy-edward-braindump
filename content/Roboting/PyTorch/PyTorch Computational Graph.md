Following [[Multivariate Chain Rule]], we end up with a large "Chain" of gradients that we have to determine. 

This can be tiresome if you do it by hand, which is why people follow a **Derive as you Go** pattern.

# Setup
We saw from previously that the chain rule in Multivariate Calculus contains some annoying summations and component-wise analysis that makes it hard to derive gradients in one shot.

Suppose we have: $$L = L(f_3(f_2(f_1(\theta))))$$
Where:
- $\theta$ is your parameter
- $f_1(\theta)$ is the first intermediate quantity
- $f_2(f_1(\theta))$ is the second intermediate quantity
- $f_3(f_2(f_1(\theta)))$ is the third intermediate quantity
- $L$ is the final scalar

We apply chain rule repeatedly, but as separate summations

**Step 1**: From $f_3$ to $f_2$ $$\frac{\partial L}{\partial (f_2)_{\beta}} = \sum_{\gamma} \frac{\partial L}{\partial (f_3)_{\gamma}} \cdot \frac{\partial (f_3)_{\gamma}}{\partial (f_2)_{\beta}}$$
**Step 2**: From $f_2$ to $f_1$ $$\frac{\partial L}{\partial (f_1)_{\alpha}} = \sum_{\beta} \frac{\partial L}{\partial (f_2)_{\beta}} \cdot \frac{\partial (f_2)_{\beta}}{\partial (f_1)_{\alpha}}$$
**Step 3**: From $f_1$ to $\theta$ $$\frac{\partial L}{\partial \theta_{\text{indices}}} = \sum_{\alpha} \frac{\partial L}{\partial (f_1)_{\alpha}} \cdot \frac{\partial (f_1)_{\alpha}}{\partial \theta_{\text{indices}}}$$
If you substitute Step 1 into Step 2 into Step 3:

$$\frac{\partial L}{\partial \theta_{\text{indices}}} = \sum_{\alpha} \sum_{\beta} \sum_{\gamma} \frac{\partial L}{\partial (f_3)_{\gamma}} \cdot \frac{\partial (f_3)_{\gamma}}{\partial (f_2)_{\beta}} \cdot \frac{\partial (f_2)_{\beta}}{\partial (f_1)_{\alpha}} \cdot \frac{\partial (f_1)_{\alpha}}{\partial \theta_{\text{indices}}}$$
# How to Computational Graph

## The Key Insight

Instead of deriving the full nested sum all at once, we can **build a graph structure** where:

- **Nodes** represent intermediate values ($\theta, f_1, f_2, f_3, L$)
- **Edges** represent dependencies (how outputs depend on inputs)

Then we compute gradients by **traversing the graph backwards**.

## Doing it by hand
Given a network architecture:
- Sketch the forward pass
- Begin at Loss
- Compute Layer Gradients
	- Compute **local gradients**
	- Apply chain rule
	- pass the input gradient up to the next layer
- Repeat until you get to the top



## Forward Pass: Build the Graph

As we compute the forward pass, we:

1. Store each intermediate value
2. Record the operations used to create them
3. Build edges showing what depends on what

**Example**: $L = L(f_3(f_2(f_1(\theta))))$

```
θ → f₁ → f₂ → f₃ → L
```

## Backward Pass: Traverse the Graph

Starting from $L$, we propagate gradients backwards through each edge:

**Step 1**: Initialize $$\frac{\partial L}{\partial L} = 1$$

**Step 2**: From $L$ to $f_3$ $$\frac{\partial L}{\partial f_3} = \frac{\partial L}{\partial L} \cdot \frac{\partial L}{\partial f_3} = 1 \cdot \frac{\partial L}{\partial f_3}$$

**Step 3**: From $f_3$ to $f_2$ (using chain rule) $$\frac{\partial L}{\partial f_2} = \frac{\partial L}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2}$$

**Step 4**: From $f_2$ to $f_1$ $$\frac{\partial L}{\partial f_1} = \frac{\partial L}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1}$$

**Step 5**: From $f_1$ to $\theta$ $$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial f_1} \cdot \frac{\partial f_1}{\partial \theta}$$

At each step, we:

1. **Receive** gradient from the next layer: $\frac{\partial L}{\partial f_{i+1}}$
2. **Compute** local gradient: $\frac{\partial f_{i+1}}{\partial f_i}$
3. **Apply chain rule**: $\frac{\partial L}{\partial f_i} = \frac{\partial L}{\partial f_{i+1}} \cdot \frac{\partial f_{i+1}}{\partial f_i}$
4. **Pass** gradient to previous layer

## Branching: When Multiple Paths Exist

If a node has **multiple children** (is used in multiple places), we sum the gradients from all paths:

```
     ┌→ f₂ →┐
θ → f₁       ├→ L
     └→ f₃ →┘
```

$$\frac{\partial L}{\partial f_1} = \frac{\partial L}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} + \frac{\partial L}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_1}$$

This automatically handles the summation in the chain rule!

## Node Types in the Graph

### 1. **Leaf Nodes** (Parameters)

- Nodes like $\theta$ (weights, biases)
- **Require gradients** - these are what we want to update
- `.requires_grad = True` in PyTorch

### 2. **Intermediate Nodes** (Activations)

- Nodes like $f_1, f_2, f_3$
- Store values during forward pass
- Compute and pass gradients during backward pass
- Can be freed after backward pass to save memory

### 3. **Output Node** (Loss)

- Node like $L$
- **Starting point** for backward pass
- Always has gradient = 1

## Operations Store Local Gradients

Each operation in the graph knows how to compute its **local gradient**:

|Operation|Forward|Backward (local gradient)|
|---|---|---|
|$y = Wx$|$y = Wx$|$\frac{\partial y}{\partial W} = x^T$, $\frac{\partial y}{\partial x} = W^T$|
|$y = \sigma(x)$|$y = \sigma(x)$|$\frac{\partial y}{\partial x} = \sigma'(x)$|
|$y = x_1 + x_2$|$y = x_1 + x_2$|$\frac{\partial y}{\partial x_1} = 1$, $\frac{\partial y}{\partial x_2} = 1$|
|$y = x_1 \odot x_2$|$y = x_1 \odot x_2$|$\frac{\partial y}{\partial x_1} = x_2$, $\frac{\partial y}{\partial x_2} = x_1$|
