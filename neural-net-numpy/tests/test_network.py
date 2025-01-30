import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import NeuralNetwork, Dense, Activation

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.network = NeuralNetwork()
        self.network.add(Dense(784, 128))  # Input layer to hidden layer
        self.network.add(Dense(128, 10))   # Hidden layer to output layer

    def test_forward_pass(self):
        # Input shape: (batch_size, input_features)
        input_data = np.random.rand(1, 784)
        output = self.network.forward(input_data)
        self.assertEqual(output.shape, (1, 10))

    def test_backpropagation(self):
        input_data = np.random.rand(1, 784)
        target = np.zeros((1, 10))
        target[0, 0] = 1  # One-hot encoded target
        
        # Forward pass
        output = self.network.forward(input_data)
        self.assertEqual(output.shape, (1, 10))
        
        # Backward pass
        gradient = output - target
        for layer in reversed(self.network.layers):
            gradient = layer.backward(gradient, learning_rate=0.01)
            
        self.assertEqual(gradient.shape, (1, 784))

    def test_add_layer(self):
        initial_layers = len(self.network.layers)
        self.network.add(Dense(10, 5))  # Add a new layer
        self.assertEqual(len(self.network.layers), initial_layers + 1)  # Check if layer count increased

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.dense = Dense(3, 2)  # 3 inputs, 2 outputs
        # Change input shape to (batch_size, features)
        self.input_data = np.array([[1, 2, 3]])  # (1,3) shape

    def test_dense_forward(self):
        output = self.dense.forward(self.input_data)
        self.assertEqual(output.shape, (1, 2))  # (batch_size, output_features)

    def test_dense_backward(self):
        output = self.dense.forward(self.input_data)
        gradient = np.array([[0.1, 0.2]])  # (batch_size, output_features)
        backward_output = self.dense.backward(gradient, 0.1)
        self.assertEqual(backward_output.shape, (1, 3))  # (batch_size, input_features)

    def test_relu_activation(self):
        x = np.array([-1, 0, 1])
        output = Activation.relu(x)
        np.testing.assert_array_equal(output, [0, 0, 1])

    def test_softmax_activation(self):
        x = np.array([[1.0, 2.0, 3.0]])  # (1,3) shape
        output = Activation.softmax(x)
        # Test row-wise normalization
        self.assertTrue(np.allclose(np.sum(output, axis=1), 1.0))

if __name__ == '__main__':
    unittest.main()