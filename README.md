# 📘 Deep Learning Models and Representation Learning

This repository is a collection of **deep learning models and experiments** covering core neural network architectures, representation learning techniques, and sequence modeling methods. The codebase is organized by model family, with each directory focusing on a specific concept or learning paradigm. The implementations are designed for **learning, experimentation, and research-oriented exploration**.

Most of these implementations originate from coursework and projects for the **Deep Learning** course at **Sharif University of Technology**.

---

## [CNN](CNN/)
- ResNet-50 classifiers for CIFAR-10 airplanes and automobiles, trained with cross-entropy, triplet loss, or both.
- A CNN with modulated deformable convolutions compared with a standard CNN on rescaled MNIST digits.
- A two-stream residual network that recognizes rotation, displacement, and scaling applied to ADE20K images.

## [FFA](FFA/)
A fully connected network trained on MNIST with the Forward-Forward algorithm; the folder for the unsupervised variant has no notebook.

## [Fully_conected](Fully_conected/)
Fully connected networks for CIFAR-10 built from layers with explicit backward passes, with comparisons of SGD, momentum, RMSProp, Adam, and dropout.

## [LSTM](LSTM/)
- Fully connected, LSTM, and bidirectional LSTM classifiers for blood-brain barrier penetration from SMILES strings (BBBP).
- Fine-tuning of a Persian GPT-2 model to generate the second hemistich of Ferdowsi couplets.

## [PCA](PCA/)
PCA on MNIST, with reconstructions and logistic-regression accuracy as a function of the number of components.

## [VAE](VAE/)
- A VAE and a conditional VAE with fully connected layers on MNIST, with t-SNE plots of their latent spaces.
- A VQ-VAE on MNIST and on colored MNIST with varying codebook sizes and dimensions.
