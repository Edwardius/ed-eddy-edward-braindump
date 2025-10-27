PyTorch has a purpose-built object to deal with large tensor data. Its `torch.Tensor` which contains the data of a uniform datatype.

# Copy / Conversion Instantiation 

```python
# Instantiation from an array (Copies array)
A = torch.Tensor([[0, 0, 0], [1, 2, 3]])

# Instantiation form a Numpy Array
B = np.ones(size=(3,3))
B_as_tensor = torch.as_tensor(B)
```

# Instantiation with Creation Ops

```python

```