"""
MNIST Neural Network Training Script

This script implements a neural network trained on MNIST using backpropagation.
Features both interactive and HTML-based visualizations of the training process.

Components:
1. Data Processing
   - MNIST dataset loading
   - Image flattening (28x28 -> 784)
   - Normalization (0-255 -> 0-1)
   - One-hot encoding of labels

2. Network Architecture
   - Input layer: 784 neurons (28x28 pixels)
   - Hidden layer: 128 neurons, ReLU activation
   - Output layer: 10 neurons, Softmax activation

3. Training Process
   - Mini-batch gradient descent
   - Cross-entropy loss
   - Batch size: 32
   - Learning rate: 0.01
   - Epochs: 100

4. Visualization
   - Interactive matplotlib dashboard
     - Weight matrices
     - Activation patterns
     - Training loss
   - HTML report
     - Training metrics
     - Network structure
     - Interactive plots

Usage:
    python src/train.py

Results:
    Visualizations saved in src/results/
    - Interactive plots update in real-time
    - HTML dashboard shows training progression
"""

import numpy as np
from keras.datasets import mnist
from network import Dense, Activation
from visualize import NetworkVisualizer
from html_visualizer import HTMLVisualizer
import os
import shutil
from tqdm import tqdm
from PIL import Image
import glob

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
X = x_train.reshape(-1, 784) / 255.0
X = X[:1000]  # Subset for testing

# One-hot encode
Y = np.zeros((1000, 10))
for i, y in enumerate(y_train[:1000]):
    Y[i, y] = 1

# Setup single results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR)

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
visualizer = NetworkVisualizer(network, RESULTS_DIR)
html_viz = HTMLVisualizer(network, RESULTS_DIR)
hidden_states = []

# Training loop
for epoch in tqdm(range(epochs), desc='Training'):
    error = 0
    hidden_states = []
    
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
        loss = error/len(X)
        tqdm.write(f'Epoch {epoch:3d} | Loss: {loss:.6f}')
        visualizer.update(epoch, loss, X[0:1], hidden_states)
        html_viz.update(epoch, loss, network[0].weights, hidden_states[-1])

# After training loop
def create_gif():
    frames = []
    imgs = glob.glob("src/results/frame_*.png")
    imgs.sort()
    for i in imgs:
        frames.append(Image.open(i))
    frames[0].save(
        "src/results/training_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )