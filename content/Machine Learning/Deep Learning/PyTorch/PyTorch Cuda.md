```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# to count the number of GPUs
torch.cuda.device_count()
```

**Output:**
```
1
```

## Placing things into the GPU
```python
tensor = torch.tensor([1, 2, 3])
tensor_on_gpu = tensor.to(device) # to move to GPU
print(tensor_on_gpu)

tensor_back_to_cpu = tensor_on_gpu.cpu()
tensor_back_to_cpu
```

**Output:**
```
tensor([1, 2, 3], device='cuda:0')
```

**Output:**
```
tensor([1, 2, 3])
```

