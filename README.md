# 📘 Deep Learning Models and Representation Learning

This repository is a collection of **deep learning models and experiments** covering core neural network architectures, representation learning techniques, and sequence modeling methods. The codebase is organized by model family, with each directory focusing on a specific concept or learning paradigm. The implementations are designed for **learning, experimentation, and research-oriented exploration**.

Most of these implementations originate from coursework and projects for the **Deep Learning** course at **Sharif University of Technology**.

---

## 📂 Repository Structure

### **CNN (Convolutional Neural Networks)**
Experiments and models based on convolutional architectures for feature learning and visual tasks:
- **CNN_Triplet_Loss_Classification** – Metric learning using triplet loss for robust classification.
- **Deformable_Convolution** – Adaptive convolutional layers for handling geometric variations.
- **Displacement_Detection** – CNN-based detection of spatial displacement patterns.

---

### **FFA (Feature-Field Alignment)**
Methods for aligning feature representations across domains or distributions:
- **Supervised** – Feature alignment using labeled data.
- **UnSupervised** – Distribution alignment without explicit supervision.

---

### **Fully_connected**
Implementations of classical **fully connected (dense) neural networks**, serving as baselines and foundations for deeper architectures.

---

### **LSTM (Long Short-Term Memory Networks)**
Sequence modeling and temporal learning tasks:
- **FC_LSTM_BiLSTM** – Comparison of fully connected, LSTM, and bidirectional LSTM architectures.
- **Poetry_Generation** – Text generation using recurrent neural networks.

---

### **PCA (Principal Component Analysis)**
Dimensionality reduction and feature extraction using classical linear techniques.

---

### **VAE (Variational Autoencoders)**
Probabilistic generative models for representation learning:
- **VAE_CVAE.ipynb** – Variational Autoencoder and Conditional VAE implementations.
- **VQ_VAE.ipynb** – Vector Quantized VAE for discrete latent representations.

---

## 🎯 Purpose
This repository serves as:
- A **learning resource** for deep learning fundamentals  
- A **research playground** for experimenting with architectures and loss functions  
- A **modular reference** for CNNs, RNNs, and generative models  

Each folder is self-contained and focuses on a specific modeling idea, making the repository easy to extend and maintain.

---

## 🛠️ Notes
- Most experiments are implemented in **Python** using common deep learning frameworks.
- Notebooks (`.ipynb`) include explanations and visualizations where appropriate.
- The repository is actively evolving as new models and ideas are explored.
