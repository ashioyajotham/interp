"""
MNIST Neural Network Training Script

This script implements a basic neural network training loop using backpropagation.
It uses the MNIST dataset to demonstrate the network's ability to learn digit classification.

Key Components:
- Data preprocessing: Flattening and normalizing MNIST images
- Network architecture: 784->128->10 neurons
- Training loop: Mini-batch gradient descent with backpropagation
"""

import numpy as np
from keras.datasets import mnist
from network import Dense, Activation
from visualize import NetworkVisualizer

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
X = x_train.reshape(-1, 784) / 255.0
X = X[:1000]  # Subset for testing

# One-hot encode
Y = np.zeros((1000, 10))
for i, y in enumerate(y_train[:1000]):
    Y[i, y] = 1

# Network
network = [
    Dense(784, 128),
    Dense(128, 10)
]

# Training
epochs = 100
learning_rate = 0.01
batch_size = 32

# Update training loop
visualizer = NetworkVisualizer(network)
hidden_states = []

for epoch in range(epochs):
    error = 0
    # Mini-batch training
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_Y = Y[i:i+batch_size]
        
        # Forward pass with activation collection
        output = batch_X
        current_states = []
        for layer in network:
            output = Activation.relu(layer.forward(output))
            current_states.append(output)
        output = Activation.softmax(output)
        
        hidden_states = current_states
        
        # Loss
        error -= np.mean(np.sum(batch_Y * np.log(output + 1e-15), axis=1))
        
        # Backward
        grad = output - batch_Y
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            
    if epoch % 5 == 0:
        visualizer.update(epoch, error/len(X), X[0:1], hidden_states)