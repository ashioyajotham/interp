# Backpropagation Implementation with NumPy

## Overview
A from-scratch implementation of backpropagation using only NumPy. Built to understand the fundamentals of neural networks by implementing gradient descent and the chain rule.

## Features
- Dense layers with customizable units
- ReLU and Softmax activations
- Mini-batch gradient descent
- Cross-entropy loss
- MNIST dataset training example

## Core Components
```python
# Key mathematical operations implemented:
forward_pass = inputs @ weights + bias
backward_pass = gradient @ weights.T  # Chain rule
weight_updates = learning_rate * input.T @ gradient
```