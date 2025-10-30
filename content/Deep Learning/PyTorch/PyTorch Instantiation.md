```python
import torch
import numpy as np
```

# Pytorch Instantiation

PyTorch has a purpose-built object to deal with large tensor data. Its `torch.Tensor` which contains the data of a uniform datatype.

A lot of the ways to build a tensor are defined by various creation operations. These are some common ones.
```python
# tensor - this is the most straight forward, copies the data you feed it. NO AUTOGRAD HISTORY
A = torch.tensor([[1, 1, 1], [1, 1, 1]])

B = np.ones(shape=(3,3))
B_copy = torch.tensor(B) # <--- this creates a copy

# random tensors
C = torch.rand(size=(6, 6))

# zeros and ones
D = torch.zeros(size=(4,5,6))
E = torch.ones(size=(5,6))

# creating a range
F = torch.arange(0, 100, 2) # START, STOP, STEP

# creating a tensor of the same shape using LIKE
F_zeros = torch.zeros_like(F)
F_ones = torch.ones_like(F)
```

## Predictable Randomness with Seed
```python
RANDOM_SEED = 88
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_A = torch.rand(3,4)
torch.random.manual_seed(seed=RANDOM_SEED)
random_tensor_B = torch.rand(3,4)

print(f"Tensor A {random_tensor_A}")
print(f"Tensor B {random_tensor_B}")
random_tensor_A == random_tensor_B
```

**Output:**
```
Tensor A tensor([[0.7731, 0.6937, 0.8303, 0.4142],
        [0.2554, 0.0190, 0.4490, 0.8893],
        [0.1977, 0.2397, 0.8601, 0.3128]])
Tensor B tensor([[0.7731, 0.6937, 0.8303, 0.4142],
        [0.2554, 0.0190, 0.4490, 0.8893],
        [0.1977, 0.2397, 0.8601, 0.3128]])
```

**Output:**
```
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])
```

```python
import torch
torch.cuda.is_available()
```

**Output:**
```
True
```

