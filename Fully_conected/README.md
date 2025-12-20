
# Notebook overview
The notebook walks through the full pipeline of: data loading and preprocessing, model definition, training, and evaluation for supervised learning tasks with fully connected networks. It is organized as an educational, experiment-friendly workflow combining explanations with modular PyTorch code. 

## Main features
Data preparation: Creating PyTorch Dataset and DataLoader objects, train/validation/test splits, and basic normalization for tabular inputs.

Model definition: Implementing configurable fully connected architectures with multiple hidden layers, nonlinear activations, and task-specific output layers. 

Training loop: Forward pass, loss computation, backpropagation, optimizer updates, and epoch-based training using SGD/Adam-style optimizers.

Evaluation: Tracking training/validation metrics over epochs and reporting final test performance (e.g., accuracy or regression error).

Experiments: Modifying depth, width, and regularization to study their effect on performance and overfitting, with plots of learning curves.

## How to run
Open fully_connected_networks.ipynb in Jupyter Lab, Jupyter Notebook, or VS Code.

Install the required Python dependencies (mainly PyTorch and standard scientific Python packages).

Run the cells sequentially to reproduce the training and evaluation results, then adjust hyperparameters or architecture definitions to run your own experiments.
