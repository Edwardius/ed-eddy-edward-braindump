# PyTorch Broadcasting

**What is broadcasting?** Automatically makes tensors compatible for various operations by implicitly expanding dimensions of size 1.

**How does it work underthehood?** When the operation is "expanded" the operand with a size of 1 gets referenced against all of the other elements in the other operand of dim size greater than 1.

**When are two tensors broadcastable?** Lining up the two tensors from right to left, two tensors are broadcastable when:
- Both dimensions are the same size
- One or both of the dimensions are of size 1 or non-existent (when non-existent, a new dimension is implicitly made to enable broadcasting to occur)

```python
import torch

a = torch.rand((2, 3, 4))
b = torch.rand((2, 3, 4))

a + b ## compatible since they are the same

a = torch.rand((1, 4, 5))
b = torch.rand((9, 4, 5))

(a * b).shape # compatible, shape is [9, 4, 5]

a = torch.rand((1, 8, 1))
b = torch.rand((12, 10, 1, 3))

(a / b).shape

# 0 means an empty dim, all other numbers in the shape are hypothetical (if it wasn't 0, this dim would be size blah)
a = torch.rand((0, 1, 0)) # 0 elements on first axis, 1 on second (if dims weren't 0), 0 elements on thirds axis
b = torch.rand((1, 2, 1))

(a * b).shape
```

**Output:**
```
torch.Size([0, 2, 0])
```

**If two tensors are broadcastable** they are calculated as follows:
- if the number of dims is different, then the tensor with the smallest dim is prepended with a dim of 1
- for each dimension, the resultant dimension is the max of the two dims of each of the tensors (1 > 5 or 3 > 1 but different numbers other than 0 and 1 are incompatible) **BUT**
- A dim of 0 results in that dim being 0 though

## Small Caveat (in place semantics)
In place operations do not allow the operand that is being operated on inplace to be broadcasted
```python
a = torch.rand((3, 3, 7))
b = torch.rand((3, 1, 7))
print(a.add_(b).shape)

a = torch.rand((1, 3, 1))
b = torch.rand((5, 4, 3, 4))
a.add_(b)
```

**Output:**
```
torch.Size([3, 3, 7])
```

**Error:**
```
[31m---------------------------------------------------------------------------[39m
[31mRuntimeError[39m                              Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[14][39m[32m, line 7[39m
[32m      5[39m a = torch.rand(([32m1[39m, [32m3[39m, [32m1[39m))
[32m      6[39m b = torch.rand(([32m5[39m, [32m4[39m, [32m3[39m, [32m4[39m))
[32m----> [39m[32m7[39m [43ma[49m[43m.[49m[43madd_[49m[43m([49m[43mb[49m[43m)[49m

[31mRuntimeError[39m: output with shape [1, 3, 1] doesn't match the broadcast shape [5, 4, 3, 4]
```

# Broadcasting in other contexts in PyTorch

## Loss Function
They generally allow for the following patterns (because they implicitly do some broadcasting logic underneath)

Simple:
```
Input: [batch_size, ...] Target: [batch_size, ...]
```

For Classification:
```
Input: [batch_size, num_classes] Target: [batch_size] <---- because the target can just be a class index not a one hot
```

For Spatial/Sequence: (for something like semantic segmentation where each pixel has a class)
```
Input: [batch_size, num_classes, d1, d2, d3, ...] Target: [batch_size, d1, d2, d3, ...] <--- because loss will be calculated per pixel (target is the class we want) and then per batch and summed
```
# Broadcasting during Matrix Multiplication

The 2 right most dims are used for matrix multiplication, the rest are broadcasted.

M x N @ N x O -> M x O
```python
a = torch.rand((1, 1, 8, 9))
b = torch.rand((2, 3, 9, 7))

(a @ b).shape # [2, 3, 8, 7]
```

**Output:**
```
torch.Size([2, 3, 8, 7])
```

# Broadcasting Puzzles

These are all a bunch of puzzles so that I can get the hang of broadcasting and tensor manipulations.
```python
''' Sequence Mask

    This one is a weird one. Say you have a matrix of sequences:
    [ 2, 3, 4, 0, 0 ]
    [ 1, 0, 0, 2, 3 ]
    [ 0, 5, 6, 7, 8 ]

    We want to know which parts of this sequence is actually part of the sequence and which is not.
    For that, we have a mask.

    Lengths: [3, 1, 5] which tells me that I only want the first 3, first, and first 5 values of
    each sequence. The rest can be ignored.
'''
A = torch.tensor(
  [[ 2, 3, 4, 0, 0 ],
  [ 1, 0, 0, 2, 3 ],
  [ 0, 5, 6, 7, 8 ],]
)
lengths = torch.tensor([3, 1, 5])

# We can do some clever broadcasting to build a mask
mask = torch.arange(A.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
''' What this did
    torch.arange(A.shape[1]).unsqueeze(0) makes a SINGLE vector
    [[0, 1, 2, 3, 4]] of shape(1, 5)
    lengths.unsqueeze(1) makes vector in the other direction
    [[3], [1], [5]] of shape(3, 1)

    the < operator then broadcasts these two tensors onto each other
    ((BROADCAST OF [1, 5]))
    [[0, 1, 2, 3, 4] < [3] ? ] 
    [[0, 1, 2, 3, 4] < [1] ? ]
    [[0, 1, 2, 3, 4] < [5] ? ]

    which expands further to 
    ((BROADCAST OF [3, 1]))
    [[0 < [3], 1 < [3], 2 < [3], 3 < [3], 4 < [3]] ? ] 
    [[0 < [1], 1 < [1], 2 < [1], 3 < [1], 4 < [1]] ? ] 
    [[0 < [5], 1 < [5], 2 < [5], 3 < [5], 4 < [5]] ? ] 

    which ends up being
    tensor([[ True,  True,  True, False, False],
        [ True, False, False, False, False],
        [ True,  True,  True,  True,  True]])
'''
A, mask, A*mask
```

**Output:**
```
(tensor([[2, 3, 4, 0, 0],
         [1, 0, 0, 2, 3],
         [0, 5, 6, 7, 8]]),
 tensor([[ True,  True,  True, False, False],
         [ True, False, False, False, False],
         [ True,  True,  True,  True,  True]]),
 tensor([[2, 3, 4, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 5, 6, 7, 8]]))
```

