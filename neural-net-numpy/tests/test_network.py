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
        input_data = np.random.rand(784, 1)  # Random input for testing
        output = self.network.forward(input_data)
        self.assertEqual(output.shape, (10, 1))  # Output should be of shape (10, 1)

    def test_backpropagation(self):
        input_data = np.random.rand(1, 784)
        target = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])  # One-hot encoded target
        output = self.network.forward(input_data)
        loss = self.network.backward(target)
        self.assertIsNotNone(loss)  # Loss should be calculated

    def test_add_layer(self):
        initial_layers = len(self.network.layers)
        self.network.add(Dense(10, 5))  # Add a new layer
        self.assertEqual(len(self.network.layers), initial_layers + 1)  # Check if layer count increased

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.dense = Dense(3, 2)  # Test with small dimensions
        self.input_data = np.array([[1], [2], [3]])

    def test_dense_forward(self):
        output = self.dense.forward(self.input_data)
        self.assertEqual(output.shape, (2, 1))

    def test_dense_backward(self):
        output = self.dense.forward(self.input_data)
        gradient = np.array([[0.1], [0.2]])
        backward_output = self.dense.backward(gradient, 0.1)
        self.assertEqual(backward_output.shape, (3, 1))

    def test_relu_activation(self):
        x = np.array([-1, 0, 1])
        output = Activation.relu(x)
        np.testing.assert_array_equal(output, [0, 0, 1])

    def test_softmax_activation(self):
        x = np.array([[1], [2], [3]])
        output = Activation.softmax(x)
        self.assertTrue(np.isclose(np.sum(output), 1))
        self.assertEqual(output.shape, (3, 1))

if __name__ == '__main__':
    unittest.main()