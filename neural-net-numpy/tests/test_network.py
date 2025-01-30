import unittest
import numpy as np
from src.network import NeuralNetwork
from src.layers import Dense, Activation

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.network = NeuralNetwork()
        self.network.add(Dense(784, 128))  # Input layer to hidden layer
        self.network.add(Activation('relu'))
        self.network.add(Dense(128, 10))   # Hidden layer to output layer
        self.network.add(Activation('softmax'))

    def test_forward_pass(self):
        input_data = np.random.rand(1, 784)  # Random input for testing
        output = self.network.forward(input_data)
        self.assertEqual(output.shape, (1, 10))  # Output should be of shape (1, 10)

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

if __name__ == '__main__':
    unittest.main()