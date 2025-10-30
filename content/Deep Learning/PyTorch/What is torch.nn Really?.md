# What is torch.nn

Just an exploration on what PyTorch actually is and the preface for each module actually existing.

## MNIST Setup
The document I am referencing is https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
it uses the MNIST Dataset to make its point.
```python
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
  content = requests.get(URL + FILENAME).content
  (PATH / FILENAME).open("wb").write(content)
```

```python
import pickle
import gzip

# The dataset is in pickle format, which is a python serializable large file format
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```

```python
from matplotlib import pyplot
import numpy as np

# Dataset is just a bunch of 28x28 pixel written numbers, just basic of the basic OCR
pyplot.imshow(x_train[88].reshape((28, 28)), cmap="gray")
pyplot.show()
```

**Output:**
```
<Figure size 640x480 with 1 Axes>
```

![[Pasted image 20251030152743.png]]

```python
import torch
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# PyTorch uses a special datatype for operations, hence we convert the data to tensor
x_train, y_train, x_valid, y_valid = map(
  partial(torch.as_tensor, device=device), 
  (x_train, y_train, x_valid, y_valid)
  )
print(x_train.shape, x_train.device)
print(y_train.shape, y_train.device)
print(x_valid.shape, x_valid.device)
print(y_valid.shape, y_valid.device)
```

**Output:**
```
torch.Size([50000, 784]) cuda:0
torch.Size([50000]) cuda:0
torch.Size([10000, 784]) cuda:0
torch.Size([10000]) cuda:0
```

## Gradients

One powerful thing about PyTorch is that it can calculate gradients automatically. This is done with `requires_grad=True`.
```python
import math

# "It works because It works" type initialization of model weights(called an Xavier Initialization)
# Well actually it has some statistical backing, to limit the chances of exploding and vanishing gradients,
# we keep te variance in activation and gradient the same
weights = torch.randn(size=(784, 10), device=device) / math.sqrt(784)  # one shot MNIST
weights.requires_grad_() # this needs to be done because / math.sqrt(784) made a new tensor!!!!!
bias = torch.zeros(10, device=device, requires_grad=True)
```

## Small Aside on Xavier Initialization

**Problem:** Keep variance constant across layers during forward/backward pass.

**Forward pass:** For $y = \sum_{i=1}^{n_{in}} w_i x_i$, assuming independence and zero mean:
$$
\text{Var}(y) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

**Goal:** $\text{Var}(y) = \text{Var}(x)$ requires:
$$
n_{in} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{in}}
$$

**Backward pass:** By symmetry, preserving gradient variance requires:
$$
n_{out} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{out}}
$$

**Xavier compromise** (average both constraints):
$$
\text{Var}(w) = \frac{2}{n_{in} + n_{out}}
$$

**So if we initialize weights as normal distribution:**
$$
W_{ij} \sim \mathcal{N}\left(\mu = 0, \sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}\right)
$$

**Implementation:**
```python
std = np.sqrt(2.0 / (n_in + n_out))
W = np.random.normal(0, std, size=(n_out, n_in))
```

But thing is the input is usually >> than the output, so people just do 
$$
\frac{1}{\sqrt{n_{in}}}
$$
```python
# Lets define some raw python functions on these tensors
def log_softmax(x):
  return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(x_batch):
  return log_softmax(x_batch @ weights + bias)
```

This is a single layer model that takes in all the input pixels, and outputs a value for the 10 classes (which  a softmax is then applied to get a final classification score). 
```python
batch_size = 64

x_batch = x_train[:batch_size]
preds = model(x_batch)
preds[0], preds.shape
```

**Output:**
```
(tensor([-2.6786, -2.4870, -2.2257, -2.0464, -2.1061, -2.3727, -1.7868, -2.7828,
         -2.5606, -2.4143], device='cuda:0', grad_fn=<SelectBackward0>),
 torch.Size([64, 10]))
```

```python
# implement negative log-likelihood
def nll(input, target):
  return -input[range(target.shape[0]), target].mean()

loss_func = nll

y_batch = y_train[:batch_size]
print(loss_func(preds, y_batch))
```

**Output:**
```
tensor(2.3618, device='cuda:0', grad_fn=<NegBackward0>)
```

```python
# implement accuracy
def accuracy(out, y_batch):
  preds = torch.argmax(out, dim=1)
  return (preds == y_batch).float().mean()

print(accuracy(preds, y_batch))
```

**Output:**
```
tensor(0.1250, device='cuda:0')
```

Using what has been written from scratch, we can build a training loop
```python
# Training loop
lr = 0.5
epochs = 2

n, c = x_train.shape

training_loss = []
for e in range(epochs):
  for i in range((n - 1) // batch_size + 1):
    # Build a batch
    start_i = i*batch_size
    end_i = start_i + batch_size
    x_batch = x_train[start_i:end_i]
    y_batch = y_train[start_i:end_i]

    # Run forward
    pred = model(x_batch)
    loss = loss_func(pred, y_batch)

    loss.backward()
    
    # Update using .data attribute
    with torch.no_grad():
      weights -= weights.grad * lr # Ok something inside Tensor Class makes this not work anymore
      bias -= bias.grad * lr
      weights.grad.zero_()
      bias.grad.zero_()

# Its a shitty model, but you can see that the loss decreased and the accuracy increased
print(loss_func(model(x_batch), y_batch), accuracy(model(x_batch), y_batch))
```

**Output:**
```
tensor(0.0651, device='cuda:0', grad_fn=<NegBackward0>) tensor(1., device='cuda:0')
```

# Refactoring the Code

The raw computation can be quite the hassle, especially if we want to build more complex networks, or share modules with other people. As a result, PyTorch abstracts alot of this raw pythonic operations away so that you can have a better time.

## `torch.nn.functional`
This contains all the functions for the `torch.nn` library. That includes a handful of activation and loss functions.
- our `negative log likelihood` function can be replaced with `F.cross_entropy`
- our `log softmax` function can be replaced with `F.cross_entropy`
```python
import torch.nn.functional as F

loss_func = F.cross_entropy

# redefine model to no longer call log_softmax
def model(x_batch):
  return x_batch @ weights + bias

print(loss_func(model(x_batch), y_batch), accuracy(model(x_batch), y_batch))
```

**Output:**
```
tensor(0.0651, device='cuda:0', grad_fn=<NllLossBackward0>) tensor(1., device='cuda:0')
```

## `nn.Module`
Speaking on model, we can abstract it away from the training loop to make things more clear and concise. `nn.Module` is a base class that lets you define the weights, bias, and forward propagation of a model. It also has a number of useful attributes like `parameters(), state_dict(), zero_grad()`.
```python
from torch import nn

class MNISTLogistic(nn.Module):
  def __init__(self, device):
    super().__init__()

    self.weights = nn.Parameter(torch.randn(size=(784, 10), device=device) / math.sqrt(784))
    self.bias = nn.Parameter(torch.zeros(10, device=device)) # bias initialized as zeros

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weights + self.bias
  
model = MNISTLogistic(device=device)
print(loss_func(model(x_batch), y_batch))
```

**Output:**
```
tensor(2.3749, device='cuda:0', grad_fn=<NllLossBackward0>)
```

Because `nn.Module` has useful attributes `.paramters()` and `.zero_grad()`, updating the weights and biases previously and manually zeroing out the gradients for each weights and biases can now be abstracted into...
```python
def fit():
  for e in range(epochs):
    for i in range((n-1) // batch_size + 1):
        # Build a batch
        start_i = i*batch_size
        end_i = start_i + batch_size
        x_batch = x_train[start_i:end_i]
        y_batch = y_train[start_i:end_i]

        # Run forward
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)

        loss.backward()

        # just update the parameters and zero the grad for the whole model
        with torch.no_grad():
          for p in model.parameters():
            p -= p.grad * lr
          model.zero_grad()

fit()    
print(loss_func(model(x_batch), y_batch))   
```

**Output:**
```
tensor(0.0654, device='cuda:0', grad_fn=<NllLossBackward0>)
```

## nn.Linear (and other layers)
Instead of manually defining weights and bias, theres a useful python class `nn.Linear` that makes one for us.

PyTorch has a bunch of other useful layers that can be used.
```python
class MNISTLogistic(nn.Module):
  def __init__(self, device=None):
    super().__init__()
    self.linear = nn.Linear(in_features=784, out_features=10, device=device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)
  
model = MNISTLogistic(device)
print(loss_func(model(x_batch), y_batch))
fit()
print(loss_func(model(x_batch), y_batch))
```

**Output:**
```
tensor(2.3214, device='cuda:0', grad_fn=<NllLossBackward0>)
tensor(0.2296, device='cuda:0', grad_fn=<NllLossBackward0>)
```

# `torch.optim`
This abstracts away optimization. We can call `step()` instead of manually updating all of our parameters with the gradient.
```python
from torch import optim

def get_model():
  model = MNISTLogistic(device)
  return model, torch.optim.SGD(model.parameters(), lr=lr) # YOU PASS IN YOUR MODEL PARAMETERS TO THE OPTIMIZER

model, opt = get_model()

for e in range(epochs):
  for i in range((n - 1) // batch_size + 1):
    # build batch
    start_i = i * batch_size
    end_i = start_i + batch_size
    x_batch = x_train[start_i:end_i]
    y_batch = y_train[start_i:end_i]

    # predict
    pred = model(x_batch)
    loss = loss_func(pred, y_batch)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(loss_func(model(x_batch), y_batch))
```

**Output:**
```
tensor(0.0829, device='cuda:0', grad_fn=<NllLossBackward0>)
```

## Dataset
PyTorch provides an abstract dataset class to manage data. TensorDataset provides a direct indexing lookup of the data.
```python
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

model, opt = get_model()

for e in range(epochs):
  for i in range((n - 1) // batch_size + 1):
    x_batch, y_batch = train_ds[i * batch_size: i*batch_size + batch_size]
    pred = model(x_batch)
    loss = loss_func(pred, y_batch)

    opt.zero_grad()
    loss.backward()
    opt.step()

print(loss_func(model(x_batch), y_batch))
```

**Output:**
```
tensor(0.0830, device='cuda:0', grad_fn=<NllLossBackward0>)
```

## DataLoader
Works in tandem with Dataset to allow for batched loading of data.
```python
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=batch_size)

for e in range(epochs):
  for x_batch, y_batch in train_dl: # very nice :)
    pred = model(x_batch)
    loss = loss_func(pred, y_batch)

    opt.zero_grad()
    loss.backward()
    opt.step()

print(loss_func(model(x_batch), y_batch))
```

**Output:**
```
tensor(0.0662, device='cuda:0', grad_fn=<NllLossBackward0>)
```

# Add Validation
There hasn't been any validation yet, so should implement it.
```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)

val_ds = TensorDataset(x_valid, y_valid)
val_dl = DataLoader(val_ds, batch_size=batch_size)


for e in range(epochs):
  model.train()
  for x_b, y_b in train_dl:
    pred = model(x_b)
    loss = loss_func(pred, y_b)

    opt.zero_grad()
    loss.backward()
    opt.step()
  
  model.eval()
  with torch.inference_mode():
    val_loss = sum(loss_func(model(x_b), y_b) for x_b, y_b in val_dl) / len(val_dl)

  print(e, val_loss)
```

**Output:**
```
0 tensor(0.2795, device='cuda:0')
1 tensor(0.2786, device='cuda:0')
```

# Create fit() and get_data()

This is more of a design decision. If you want to can abstract everything to make it like three scripts.
```python
def batch_loss(model, loss_func, x_b, y_b):
  pred = model(x_b)
  return loss_func(pred, y_b)

def get_data(train_ds, val_ds):
  return (
    DataLoader(train_ds, batch_size=batch_size, shuffle=True),
    DataLoader(val_ds, batch_size=batch_size * 2),
  )

def fit(epochs, model, loss_func, opt, train_dl, val_dl):
  for e in range(epochs):
    model.train()
    for x_b, y_b in train_dl:
      loss = batch_loss(model, loss_func, x_b, y_b)

      opt.zero_grad()
      loss.backward()
      opt.step()
    
    model.eval()
    with torch.inference_mode():
      val_loss = sum(batch_loss(model, loss_func, x_b, y_b) for x_b, y_b in val_dl) / len(val_dl)

    print(e, val_loss)

train_dl, val_dl = get_data(train_ds, val_ds)
fit(10, model, loss_func, opt, train_dl, val_dl)
```

**Output:**
```
0 tensor(0.2233, device='cuda:0')
1 tensor(0.3145, device='cuda:0')
2 tensor(0.2947, device='cuda:0')
3 tensor(0.2699, device='cuda:0')
4 tensor(0.3732, device='cuda:0')
5 tensor(0.6905, device='cuda:0')
6 tensor(0.2980, device='cuda:0')
7 tensor(0.2964, device='cuda:0')
8 tensor(0.3577, device='cuda:0')
9 tensor(0.2735, device='cuda:0')
```

# Making this thing actually work
The whole model is a single linear layer at the moment, lets actually set this model up for success. One way is to introduce convolutional layers to the network. These will learn kernels (filters) that are convoluted (rolling dot-product window) over the image to recognize patterns. These patterns and their location then provide a better classification for vision in general.
```python
class MNISTCNN(nn.Module):
  def __init__(self, device=None):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, device=device)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, device=device)
    self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1,device=device)
  
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 1, 28, 28) # -1 means "infer this dimension automatically"
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.avg_pool2d(x, 4)
    return x.view(-1, x.size(1))
  
# Momentum is a variation of SGD which is aware of past updates to help with the current update (hence momentum)
model = MNISTCNN(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

fit(10, model, loss_func, opt, train_dl, val_dl)
```

**Output:**
```
0 tensor(0.5407, device='cuda:0')
1 tensor(0.3209, device='cuda:0')
2 tensor(0.2648, device='cuda:0')
3 tensor(0.2313, device='cuda:0')
4 tensor(0.1830, device='cuda:0')
5 tensor(0.1886, device='cuda:0')
6 tensor(0.1700, device='cuda:0')
7 tensor(0.1602, device='cuda:0')
8 tensor(0.1453, device='cuda:0')
9 tensor(0.1497, device='cuda:0')
```

# `nn.Sequential`

This is just a quicker way to define a model. idk why people do this.
```python
class Lambda(nn.Module):
  def __init__(self, func, device=None):
    super().__init__()
    self.func = func

  def forward(self, x):
    return self.func(x)
  
def preprocess(x):
  return x.view(-1, 1, 28, 28)

model = nn.Sequential(
  Lambda(preprocess),
  nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, device=device),
  nn.ReLU(),
  nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, device=device),
  nn.ReLU(),
  nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, device=device),
  nn.ReLU(),
  nn.AvgPool2d(4),
  Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

fit(10, model, loss_func, opt, train_dl, val_dl)
```

**Output:**
```
0 tensor(0.5411, device='cuda:0')
1 tensor(0.3231, device='cuda:0')
2 tensor(0.2601, device='cuda:0')
3 tensor(0.2159, device='cuda:0')
4 tensor(0.2040, device='cuda:0')
5 tensor(0.1843, device='cuda:0')
6 tensor(0.1636, device='cuda:0')
7 tensor(0.1547, device='cuda:0')
8 tensor(0.1656, device='cuda:0')
9 tensor(0.1423, device='cuda:0')
```

