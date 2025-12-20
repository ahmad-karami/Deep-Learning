# Supervised Feature-Field Alignment
implementation of FFA for labelled MNIST classification.​

- Label-aware data generation

  - Construction of positive samples by embedding the correct label as a one-hot code into the first pixels of each flattened image.​

  - Construction of negative samples by generating mismatched labels and embedding them in the same way, producing label–image inconsistencies.​

- FFA layer and network

  - Definition of a custom Layer class that normalizes inputs, applies a linear + ReLU transform, and computes the per-sample energy g(x) as the mean squared activation.​

  - A Net class stacks multiple FFA layers and provides a predict method that converts energies into class predictions.​

- Training objective and results

  - Layer-wise training that minimizes energy on positive samples while maximizing (up to a margin) energy on negative samples, using an energy-difference loss with threshold.​

  - Reported MNIST performance with high train and test accuracy (around 94%), demonstrating that FFA can act as a competitive supervised classifier.
