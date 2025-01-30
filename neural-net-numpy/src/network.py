import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

class Layer:
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # He initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, input):
        self.input = input
        # Input shape: (batch_size, input_size)
        # Output shape: (batch_size, output_size)
        return np.dot(input, self.weights.T) + self.bias
        
    def backward(self, output_gradient, learning_rate):
        # output_gradient shape: (batch_size, output_size)
        weights_gradient = np.dot(output_gradient.T, self.input)
        input_gradient = np.dot(output_gradient, self.weights)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        return input_gradient

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

__all__ = ['Layer', 'Dense', 'Activation', 'NeuralNetwork']