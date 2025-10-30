# Inplace Operations

Inplace operations operate on the tensor's own data, without creating a copy.

Most operations inside a PyTorch model are not in place, this is becuase inplace operations break the autograd.

Some layer operations have conditional support for inplace (like `nn.ReLU` (inplace=True)) but its limited a implementation specific.

I assume that generally inplace operations are used during the data pre and post processing steps because there is no autograd, and copying data can be expensive.
- **another place is gradients** the optimizer updates the gradients in place, as all the layers have it.

## Common Inplace Operations
all general mathematical operations: add_, sub_, mul_, div_, pow_, log_, neg_, exp_, log_, sqrt_ 

clamping: clamp_(min, max), clamp_min_, clamp_max_ (stops values in the tensor for exceeding the clamped range)

filling: fill_, zero_

manipulation: copy_, transpose_, t_, resize_, reshape_ <--- done if possible without copy

indexing assignment

some activation functions: F.relu_ , f.dropout_ <-- randomly removes a neuron (set to 0), while keeping the expected sum of the neurons the same.


```python
import torch
from torch.functional import F
a = torch.randn(100)

F.relu_(torch.softmax(a, dim=0))
```

**Output:**
```
tensor([0.0027, 0.0121, 0.0463, 0.0105, 0.0080, 0.0029, 0.0009, 0.0071, 0.0124,
        0.0400, 0.0044, 0.0311, 0.0023, 0.0027, 0.0009, 0.0031, 0.0027, 0.0027,
        0.0044, 0.0015, 0.0014, 0.0019, 0.0339, 0.0011, 0.0079, 0.0139, 0.0070,
        0.0118, 0.0722, 0.0012, 0.0047, 0.0029, 0.0014, 0.0207, 0.0117, 0.0260,
        0.0061, 0.0017, 0.0099, 0.0008, 0.0101, 0.0036, 0.0078, 0.0104, 0.0005,
        0.0008, 0.0037, 0.0057, 0.0019, 0.0171, 0.0143, 0.0050, 0.0064, 0.0094,
        0.0194, 0.0083, 0.0101, 0.0082, 0.0011, 0.0050, 0.0042, 0.0270, 0.0009,
        0.0060, 0.0006, 0.0019, 0.0808, 0.0028, 0.0107, 0.0079, 0.0021, 0.0049,
        0.0011, 0.0039, 0.0071, 0.0053, 0.0028, 0.0124, 0.0079, 0.0069, 0.0190,
        0.0028, 0.0020, 0.0083, 0.0083, 0.0060, 0.0078, 0.0049, 0.0412, 0.0291,
        0.0008, 0.0450, 0.0031, 0.0018, 0.0109, 0.0055, 0.0016, 0.0041, 0.0097,
        0.0056])
```


