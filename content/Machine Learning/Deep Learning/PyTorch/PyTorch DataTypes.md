```python
import torch
```

# PyTorch DataTypes

`torch.tensor` can handle a number of different datatypes.
```python
float64_tensor = torch.tensor([1,1,1],dtype=torch.float64)
float64_tensor

float16_tensor = torch.rand(size=(3, 4), dtype=torch.float16)
float16_tensor
```

**Output:**
```
tensor([[0.1782, 0.5679, 0.7686, 0.1338],
        [0.4863, 0.4971, 0.0088, 0.3740],
        [0.0405, 0.4556, 0.3960, 0.5776]], dtype=torch.float16)
```

## Changing Datatypes

```python
tensor = torch.rand(size=(4,4))
tensor_float16 = tensor.type(torch.float16) # this converts the current tensor to float 16 type (but copy tho)
tensor_int8 = tensor.type(torch.int8)
tensor, tensor_float16, tensor_int8
```

**Output:**
```
(tensor([[0.8292, 0.7424, 0.2252, 0.9590],
         [0.6769, 0.2841, 0.7793, 0.3459],
         [0.5991, 0.6822, 0.2904, 0.1206],
         [0.3719, 0.9203, 0.7757, 0.0407]]),
 tensor([[0.8291, 0.7422, 0.2252, 0.9590],
         [0.6768, 0.2842, 0.7793, 0.3459],
         [0.5991, 0.6821, 0.2903, 0.1205],
         [0.3718, 0.9204, 0.7759, 0.0407]], dtype=torch.float16),
 tensor([[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]], dtype=torch.int8))
```

