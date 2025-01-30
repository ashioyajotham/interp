# Neural Network Implementation from Scratch

## Overview
A numpy implementation of a neural network with backpropagation, trained on MNIST dataset. The network is trained using mini-batch gradient descent and cross-entropy loss. The training process is visualized in an interactive dashboard.

## Features
- Dense layers with customizable units
- ReLU and Softmax activations
- Mini-batch gradient descent
- Cross-entropy loss
- MNIST dataset training example

## Training Visualization
Latest training run:
<img src="src/results/latest/training_animation.gif" width="800">

## Interactive Dashboard
<img src="src/results/latest/dashboard.png" width="800">

## Core Components
```python
# Key mathematical operations:
forward_pass = inputs @ weights + bias
backward_pass = gradient @ weights.T  # Chain rule
weight_updates = learning_rate * input.T @ gradient
```
````markdown