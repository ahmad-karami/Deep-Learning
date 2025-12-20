# UnSupervised Feature-Field Alignment
FFA ideas to a label-free setup on MNIST using spatial masks and synthetic data transformations.

- Mask generation and visualization

  - A create_mask function builds random binary masks via iterative 1D convolutions, producing smooth foreground–background partitions over the image.
  - Visualization of masks and their complements to show how different regions of the image are selected or suppressed.​

- Unsupervised data construction

  - Generation of positive and negative data by applying masks (and their complements) to MNIST digits, including examples of “negative data” where digits are corrupted or partially removed.
  - Use of large batch loaders (e.g., 5000 train, 1000 test) to build masked datasets without using digit labels.​

- Unsupervised FFA training and analysis

  - FFA-style layers trained to discriminate between different masked views using only energy statistics, without explicit class labels.
  - Qualitative inspection of learned behavior through plotted examples and intermediate outputs, highlighting how the model responds to structured vs. corrupted patterns.​



