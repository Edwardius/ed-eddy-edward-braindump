```python
import torch
```

# PyTorch Tensor Operations

## Elementwise operations
If you operate with a scalar, it will perform that operation on all of the elements.
```python
# Addition
A = torch.rand(size=(3,3))
print(A)
print(A + torch.ones(size=(3,3)))
print(A + 1) # this functionally does the same thing as above!! broadcasts
```

**Output:**
```
tensor([[0.2339, 0.7098, 0.6142],
        [0.0381, 0.8267, 0.1002],
        [0.7466, 0.1895, 0.2673]])
tensor([[1.2339, 1.7098, 1.6142],
        [1.0381, 1.8267, 1.1002],
        [1.7466, 1.1895, 1.2673]])
tensor([[1.2339, 1.7098, 1.6142],
        [1.0381, 1.8267, 1.1002],
        [1.7466, 1.1895, 1.2673]])
```

```python
# Subtraction
B = torch.rand(size=(3,3))
print(B)
print(B - 10)
```

**Output:**
```
tensor([[0.6257, 0.0488, 0.1315],
        [0.3066, 0.1508, 0.1718],
        [0.8165, 0.6892, 0.5471]])
tensor([[-9.3743, -9.9512, -9.8685],
        [-9.6934, -9.8492, -9.8282],
        [-9.1835, -9.3108, -9.4529]])
```

```python
# Multiplication
C = torch.rand(size=(2,4))
print(C)
print(C * 0)
```

**Output:**
```
tensor([[0.1088, 0.0498, 0.1934, 0.1715],
        [0.5359, 0.0798, 0.7088, 0.4292]])
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.]])
```

```python
# Division
D = torch.rand(size=(9,9))
print(D)
print(D / 10)
```

**Output:**
```
tensor([[0.2233, 0.5975, 0.7809, 0.4628, 0.5885, 0.3520, 0.2315, 0.7225, 0.3431],
        [0.6590, 0.3595, 0.5055, 0.3556, 0.0775, 0.1074, 0.0194, 0.4781, 0.5521],
        [0.6431, 0.9693, 0.8376, 0.8272, 0.6010, 0.2416, 0.0118, 0.1930, 0.3059],
        [0.5878, 0.8586, 0.7320, 0.8026, 0.4704, 0.9836, 0.9053, 0.3315, 0.7002],
        [0.1529, 0.6839, 0.4797, 0.3638, 0.4218, 0.5437, 0.5274, 0.4355, 0.0865],
        [0.2732, 0.0304, 0.9109, 0.9768, 0.4425, 0.2580, 0.4441, 0.6971, 0.9129],
        [0.8907, 0.1630, 0.9689, 0.1476, 0.7911, 0.9523, 0.8729, 0.2922, 0.4187],
        [0.0350, 0.1326, 0.6774, 0.5658, 0.4687, 0.0396, 0.7862, 0.5471, 0.4041],
        [0.3690, 0.5235, 0.5344, 0.5752, 0.6925, 0.4329, 0.3679, 0.7114, 0.3887]])
tensor([[0.0223, 0.0597, 0.0781, 0.0463, 0.0588, 0.0352, 0.0232, 0.0722, 0.0343],
        [0.0659, 0.0360, 0.0505, 0.0356, 0.0077, 0.0107, 0.0019, 0.0478, 0.0552],
        [0.0643, 0.0969, 0.0838, 0.0827, 0.0601, 0.0242, 0.0012, 0.0193, 0.0306],
        [0.0588, 0.0859, 0.0732, 0.0803, 0.0470, 0.0984, 0.0905, 0.0331, 0.0700],
        [0.0153, 0.0684, 0.0480, 0.0364, 0.0422, 0.0544, 0.0527, 0.0436, 0.0087],
        [0.0273, 0.0030, 0.0911, 0.0977, 0.0442, 0.0258, 0.0444, 0.0697, 0.0913],
        [0.0891, 0.0163, 0.0969, 0.0148, 0.0791, 0.0952, 0.0873, 0.0292, 0.0419],
        [0.0035, 0.0133, 0.0677, 0.0566, 0.0469, 0.0040, 0.0786, 0.0547, 0.0404],
        [0.0369, 0.0523, 0.0534, 0.0575, 0.0692, 0.0433, 0.0368, 0.0711, 0.0389]])
```

```python
# Diff, finds the difference between an element ahead and the current element, for each element
a = torch.arange(10)
# to keep the tensor the same size, use prepend or append, must be of compatible shape
a, torch.diff(a), torch.diff(a, prepend=torch.zeros(1, dtype=torch.long))
```

**Output:**
```
(tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
 tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
```

## Matrix Multiplication

Most common operation. **inner dimensions must match, result is the outer dimensions**

(N, M) @ (M, X) --> (X, N)
```python
a = torch.tensor([4, 5, 6])
b = torch.tensor([1, 2, 3])

# Elementwise multiplication
print(a * b)

# Proper matrix multiplication
print(a @ b)
print(torch.matmul(a,b))

# For matrix multiplcation, the innermost dimension must match!
A = torch.tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
B = torch.tensor([[3, 4, 5], [6, 7, 8]]) # (2, 3)

print(A @ B.T) # (2, 3) @ (3, 2) -> (2, 2)
print(torch.mm(A, B.T))
print(torch.mm(B.T, A)) # (3, 2) @ (2, 3) -> (3, 3)
```

**Output:**
```
tensor([ 4, 10, 18])
tensor(32)
tensor(32)
tensor([[ 26,  44],
        [ 62, 107]])
tensor([[ 26,  44],
        [ 62, 107]])
tensor([[27, 36, 45],
        [32, 43, 54],
        [37, 50, 63]])
```

```python
# Linear Layer
# y = Ax + b

torch.manual_seed(42)

linear_layer = torch.nn.Linear(in_features=3, out_features=6)

x = torch.tensor([[1, 2, 1], [1, 2, 1]], dtype=torch.float32)

output = linear_layer(x)
output
```

**Output:**
```
tensor([[0.9950, 0.5410, 0.6400, 0.6204, 0.6268, 1.2770],
        [0.9950, 0.5410, 0.6400, 0.6204, 0.6268, 1.2770]],
       grad_fn=<AddmmBackward0>)
```

```python
import torch
# Inner product
a = torch.rand(10)
b = torch.rand(10)

a.T@b, b@a.T, (a.unsqueeze(1)@b.unsqueeze(0)).shape, torch.outer(a,b).shape
```

**Output:**
```
(tensor(2.7346), tensor(2.7346), torch.Size([10, 10]), torch.Size([10, 10]))
```

## Reshaping, Stacking, Squeezing and Unsqueezing, and other "shifters"


| Method | One-line description |
| ----- | ----- |
| [`torch.reshape(input, shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape) | Reshapes `input` to `shape` (if compatible), can also use `torch.Tensor.reshape()`. |
| [`Tensor.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | Returns a **view** of the original tensor in a different `shape` but shares the same data as the original tensor. |
| [`torch.stack(tensors, dim=0)`](https://pytorch.org/docs/1.9.1/generated/torch.stack.html) | Concatenates a sequence of `tensors` along a new dimension (`dim`), all `tensors` must be same size. |
| [`torch.squeeze(input)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | Squeezes `input` to remove all the dimenions with value `1`. |
| [`torch.unsqueeze(input, dim)`](https://pytorch.org/docs/1.9.1/generated/torch.unsqueeze.html) | Returns `input` with a dimension value of `1` added at `dim`. | 
| [`torch.permute(input, dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | Returns a **view** of the original `input` with its dimensions permuted (rearranged) to `dims`. | 

Theres also other like `torch.roll` which shifts elements in a circular fashion within the tensor

Also `torch.flip` which flips the tensor across a set of dimensions.
```python
x = torch.arange(-100, 100, 2)
X = x.reshape([10, 10])
X_unsqueezed = torch.unsqueeze(X, 2)
print(X_unsqueezed.shape)

X_squeezed = torch.squeeze(X_unsqueezed)
print(X_squeezed.shape)

X_2 = X.view([10, 10])
X_stacked = torch.stack([X_2, X], dim=0)
print(X_stacked.shape)

X_permute = torch.permute(X_stacked, (1, 0, 2))
print(X_permute.shape)
```

**Output:**
```
torch.Size([10, 10, 1])
torch.Size([10, 10])
torch.Size([2, 10, 10])
torch.Size([10, 2, 10])
```

> IMPORTANT TO NOTE THAT RESHAPE CREATES A WHOLE NEW TENSOR WHILE PERMUTE JUST RETURNS A VIEW OF THE TENSOR. SO A VIEW MEANS DOES NOT OWN.
```python
X = torch.rand(size=(50, 10, 3))
X_view = X[:30:2, :8, 2:] # splicing is the same as numpy, start:stop:step
X_view.shape

```

**Output:**
```
torch.Size([15, 8, 1])
```

```python
# Torch Roll, "Shifts the elements right in a circular fashion", neg shift shifts left
A = torch.arange(10)
A, A.roll(shifts=1), A.roll(shifts=-1)
```

**Output:**
```
(tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 tensor([9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
 tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
```

```python
# Torch Flip
A = torch.arange(16).reshape(2, 2, 4)
A, A.flip(dims=[0,2])
```

**Output:**
```
(tensor([[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7]],
 
         [[ 8,  9, 10, 11],
          [12, 13, 14, 15]]]),
 tensor([[[11, 10,  9,  8],
          [15, 14, 13, 12]],
 
         [[ 3,  2,  1,  0],
          [ 7,  6,  5,  4]]]))
```

# Diagonals and Eye
```python
import torch 

# use torch.diag to get the diagonal vector (will work with non-square matricies)
a = torch.rand(3,4)
torch.diag(a)
a.diag()
```

**Output:**
```
tensor([0.3291, 0.9518, 0.7917])
```

```python
# use torch.eye to get an identity matrix, one value makes a square matrix, two makes a unsymmetrical identity matrix
b = torch.eye(10)
c = torch.eye(3, 10)
c
```

**Output:**
```
tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
```

```python
# torch.triu filters our the bottom half of a matric across the diagonal (values on the diagonal and kept)
A = torch.ones(8,8)
torch.triu(A) # Both work
A.triu() # both work
```

**Output:**
```
tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1.]])
```

# Bincount
Its a tensor that keeps track of the number of occurences of each number in a tensor (the index is the number). Only works with 1D tensor
```python
A = torch.arange(10)
print(A.bincount())

A = torch.tensor([3, 3, 4, 1, 1, 1, 1, 0, 5])
# minlength defines the minimum number of bins to have
A.bincount(minlength=20), torch.bincount(A, minlength=20)
```

**Output:**
```
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

**Output:**
```
(tensor([1, 4, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
 tensor([1, 4, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
```

# Scatter Add
Given a tensor, scatter_add Takes a src tensor of values, a tensor of "links" referring what index we should take that value, and adds those values into the tensor 
```python
A = torch.tensor([1, 1, 34, 6, 88], dtype=torch.long)
links = torch.tensor([0, 0, 8, 0, 0], dtype=torch.long)
B = torch.zeros(10, dtype=torch.long)

A, B, B.scatter_add(dim=0, index=links, src=A)
```

**Output:**
```
(tensor([ 1,  1, 34,  6, 88]),
 tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
 tensor([96,  0,  0,  0,  0,  0,  0,  0, 34,  0]))
```

# Heaviside

Returns the output of a heaviside function (values at 0 are defined by another tensor)

$$
h(x)=\begin{cases}
0 & x < 0 \\
value &  x = 0 \\
1 & x>0
\end{cases}
$$

But here values is a tensor indicating the value we want to set indexes to if they eq zero
```python
A = torch.tensor([-0.1, 0, 0, 1, 2, -2, 1, 0, 0, 0])
values = torch.tensor([10, 34, 3, 1, 1, 1, 4, 8, 7, 1], dtype=torch.float)

torch.heaviside(A, values=values)
```

**Output:**
```
tensor([ 0., 34.,  3.,  1.,  1.,  0.,  1.,  8.,  7.,  1.])
```

# Bucketize
Places values into "buckets" defined by a vector of boundaries.
- left bucket (default): values exactly on a boundary will fall into the bucket on the left
       a   b   c   d
... ___| __| __| __| __ ...  

- right bucket: values exactly on the boundary wiill fall into the bucket on the right
       a   b   c   d
... __ |__ |__ |__ |__ ...  
```python
boundaries = torch.tensor([1, 4, 8, 10])
values = torch.tensor([3, 19, 10, 5, 8])

torch.bucketize(values, boundaries), torch.bucketize(values, boundaries, right=True)
```

**Output:**
```
(tensor([1, 4, 3, 2, 2]), tensor([1, 4, 4, 2, 3]))
```

