# Convolutional Network Projects

This folder contains three PyTorch projects built on convolutional networks.

## [CNN_Triplet_Loss_Classification](CNN_Triplet_Loss_Classification/)
Three ways to train an ImageNet-pretrained ResNet-50 to tell CIFAR-10 airplanes from automobiles: a linear classifier on frozen features, triplet-loss fine-tuning followed by a classifier, and joint triplet and cross-entropy training.

## [Deformable_Convolution](Deformable_Convolution/)
A small CNN with modulated deformable convolutions compared with the same CNN with standard convolutions on MNIST digits rescaled by factors from 0.5 to 1.5.

## [Displacement_Detection](Displacement_Detection/)
A two-stream residual network with feature exchange that recognizes which combination of rotation, displacement, and scaling was applied to an ADE20K image.
