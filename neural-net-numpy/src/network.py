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
    def forward(self, input_data):
        raise NotImplementedError
        
    def backward(self, gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, input_data):
        # input: (batch_size, input_size)
        # output: (batch_size, output_size)
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias
        
    def backward(self, gradient, learning_rate):
        # gradient: (batch_size, output_size)
        weights_gradient = np.dot(self.input.T, gradient)
        input_gradient = np.dot(gradient, self.weights.T)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(gradient, axis=0, keepdims=True)
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
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

__all__ = ['Layer', 'Dense', 'Activation', 'NeuralNetwork']