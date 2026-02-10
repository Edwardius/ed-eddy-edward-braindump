These are implicit assumptions that every type of neural network layer has on the data it's dealing with.
# Convolutional Networks

- **Hierarchy** assumes that signal is made up of simpler signal primitives that can be combined together
- **Translational Equivariance** translation of patterns in the input result in a similar translation in the output
- **Parameter Sharing** the same parameters are shared across locations in the input
- **Locality** pixels beside each other are more closely related than pixels farther apart

### When to use
- Time series data (where actions in a close sequence matter more)
- Images
- 3d Data (but transformers are pretty close)

### When not to use
- Tabular data
- Data where order doesn't matter: where the order of adjacent signals dont matter
- Data where global relationships matter more than local ones

# Fully Connected Layer

- **No structure assumptions** makes no assumptions on the inherent structure of the data, any structure has to be learned
- **Global receptive field** sees all of the input at once
- **Position dependent** the position of the pattern matters, (not translational equivariant )

### When to use
- **Final classification layers** where we should consider all the features we've encoded as a whole
- **Tabular data** where there is no idea of spatial / temporal understanding of the data
- **Low-dimensional Latent spaces** like encoding data (word embeddings?)
- **Coordinate networks** networks that use coordinates an inputs and outputs some value (NeRF)

### When not to use
- **High-dimensional Latent Spaces** too many parameters
- **When translation equivariance is desired** use a conv instead

# Recurrent Neural Nets

- **Data is sequential / temporal** our understanding of the data evolves as we sequentially go through the data
- **Markovian Process** Current state depends on the past state, but not the full history (vanishing gradient as you get too far away)
- **Translational Equivariance, Temporal** translation in the sequence in time will result in a similar translation in the output of the RNN (CNNs have the same ordeal, but for both space and time)
- **Causality** can only look backwards (and we assume that we only need to look backwards in time)

### When to use
- **Sequential Data** like text, or an action through time
- **Data of variable length** RNNs can run on data sequentially, so the length of the data doesn't matter (it loses memory of the initial part of the data though)
- **Online/streaming process needed** if we need the NN to function in an online manner on a stream of data
- **Past context is needed** and future dont matter

### When not to use
- **Non-sequential data**
- **Data of fixed length**, 9/10 times transformers perform better
- **Long range dependencies**, when the RNN needs to know context that is from the start of the sequence (it will forget)
- **Data where order doesn't matter**

# Self-Attention Layers

- **Almost no assumptions** so its generalizable, but needs more data
- **Order does not matter**, inherently, the layer does not deal with order, you have to make it learn order, which can be done with position encodings
- **Data far away, close together, are equally as likely to be weighed more or less** aka global receptive field
- **Data heavily depends on its context**
- **Variable Length** the actual attention layer can handle variable lengths, but to make the training process quicker, we usually batch and pad the inputs

### When to use
- Data depends on data far away (or the entire context)
- Can work on sets and pointclouds
- when you got LOTS of data

### When not to use
- Too little data
- Strong spatial / temporal patterns exist
- Computational contraints

# Pooling Layers (MaxPool, AvgPool)

- **We can compress the data without losing too much information**
- small shifts in the data don't matter too much
- when you want to build hierarchy more explicitly
- Maxpool: most prominent feature wins
- Avgpool: should aggregate features together to get the best one

# Batch Normalization

- The statistical characteristics of the Batch are roughly similar to the statistical characteristics of the entire dataset
- normalization is needed and helps

# Layer Norm

- The statistical characteristics of the layer should be considered separately from the batch

# Dropout

- no single neuron matters