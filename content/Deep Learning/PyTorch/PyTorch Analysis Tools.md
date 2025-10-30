```python
import torch
```

# Analyzing Tensors
```python
A = torch.rand(size=(5,5,6))

# Shape returns the shape of the tensor from outer most to inner most
print(f"shape: {A.shape}")
# ndims returns the number of dims of a tensor
print(f"number of dimensions: {A.ndim}")
# dtype returns the datatype of the tensor
print(f"datatype {A.dtype}")
# device can show you what device the tensor is stored on (CPU, GPU)
print(f"device: {A.device}")
```

**Output:**
```
shape: torch.Size([5, 5, 6])
number of dimensions: 3
datatype torch.float32
device: cpu
```

