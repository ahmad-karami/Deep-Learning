# Forward-Forward Training on MNIST

This notebook trains a fully connected network on MNIST with the Forward-Forward algorithm in PyTorch, in which each layer is trained with a local objective instead of backpropagation through the whole network. The label is written into the first 10 pixels of each flattened image as a one-hot code, which gives positive samples with the correct label and negative samples with a random wrong label. Each layer normalizes its input, applies a linear map and a ReLU, and is trained with its own Adam optimizer so that its goodness, the mean squared activation, is above a threshold of 2.0 for positive samples and below it for negative samples. A network with layer sizes 784, 500, and 500 is trained layer by layer, for 1,000 steps per layer on each batch of 5,000 training images. To classify an image, the network tries all 10 labels and picks the one with the largest goodness summed over the layers, and the notebook reports the accuracy on the training and test sets.

## Results
The network reaches an accuracy of 94.44% on the training set and 94.02% on the test set. The notebook does not record the loss during training, so it has no training curves.
