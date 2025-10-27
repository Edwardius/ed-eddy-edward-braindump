https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

> Holy fucking shit what is this why is it so large, why why why why why why why why

This thing is fucking huge. And it's probably extremely important for alot of fields.

Actually, what's interesting is that matrix calculus can only get me so far. There are operations in Deep Learning that matrix calculus cannot be used on.

- **Convolution** (weight sharing + local connectivity)
- **Softmax** (nonlinear normalization with coupling)
- **BatchNorm** (normalization couples batch elements)
- **Pooling** (max, avg - reduction operations)

These operations operate on the elements themselves, which clashes with matrix calculus which deals with matrices operating on matrices with matrix-level operations.

Realistically Deep Learning evolves from empirical discovery, not pure mathematical design, so the strive of a mathematician for a perfect world like matrix calculus is hard when engineers just want to get something working.