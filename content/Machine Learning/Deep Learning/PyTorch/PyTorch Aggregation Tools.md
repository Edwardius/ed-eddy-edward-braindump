```python
import torch
```

# PyTorch Aggregation

Refers to min, max, mean, sum, prod. These are all operations that devolves the tensor down to a single element.
```python
x = torch.arange(0, 100, 10, dtype=torch.float32)
x
```

**Output:**
```
tensor([ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.])
```

```python
x.min()
x.max()
x.mean()
x.sum() # sum of all the elements in a matrix

torch.min(x)
torch.max(x)
torch.mean(x)
torch.sum(x)
torch.prod(x)

x.argmax() # finds the index of the max value
x.argmin() # finds the index of the min value
```

**Output:**
```
tensor(0)
```

