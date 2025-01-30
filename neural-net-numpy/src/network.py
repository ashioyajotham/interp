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
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((output_size, 1))
        
    def forward(self, input):
        self.input = input
        # Reshape input if needed
        if len(input.shape) == 2:
            self.input = input.T
        return (np.dot(self.weights, self.input) + self.bias).T

    def backward(self, output_gradient, learning_rate):
        if len(output_gradient.shape) == 2:
            output_gradient = output_gradient.T
            
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        
        return input_gradient.T

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