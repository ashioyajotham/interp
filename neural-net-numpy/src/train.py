import numpy as np
from keras.datasets import mnist
from network import Dense, relu, softmax

def train_neural_network():
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and reshape
    X = x_train.reshape(x_train.shape[0], 784, 1) / 255.0
    X = X[:1000] # Start with subset for testing

    # One-hot encode outputs
    Y = np.zeros((y_train[:1000].size, 10, 1))
    for i, y in enumerate(y_train[:1000]):
        Y[i][y][0] = 1

    # Create network
    network = [
        Dense(784, 128),
        Dense(128, 10)
    ]

    # Training parameters
    epochs = 100
    learning_rate = 0.1

    for epoch in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # Forward
            output = x
            for layer in network:
                output = relu(layer.forward(output))
            output = softmax(output)
            
            # Backward
            grad = output - y
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
                
            error += np.mean(np.abs(grad))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Error: {error/len(X)}')

if __name__ == "__main__":
    train_neural_network()