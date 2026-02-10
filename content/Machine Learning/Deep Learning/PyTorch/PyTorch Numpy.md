```python
import torch
import numpy as np
```

# PyTorch and Numpy

There exists conversions between them.
```python
array = np.arange(0.0, 10.0, 1.0)
tensor = torch.as_tensor(array) # this shares the ownership with array
tensor_copy = torch.from_numpy(array)
tensor.dtype
tensor
tensor[2] = 100 # changes array
array
tensor_copy[1] = 99 # changes array (but because as_tensor was called weirdly enough)
array = array + 1 # does not change tensor
tensor, array

arr = np.ones(shape=(3, 3))
arr
tensor = torch.from_numpy(arr)
tensor
tensor = tensor + 1 # here i changed the tensor but the arr was left untouched
tensor, arr
```

**Output:**
```
(tensor([[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]], dtype=torch.float64),
 array([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]]))
```

