This notebook introduces Principal Component Analysis (PCA) for data compression using eigenvalues and eigenvectors, with experiments conducted on the MNIST handwritten digit dataset. PCA is applied with varying numbers of components to study the effect of information loss.

To evaluate the impact of compression on learning performance, logistic regression is trained on PCA-processed data, and the resulting classification accuracy is analyzed as a function of the number of retained components. The notebook highlights the trade-off between data compression and model performance.
