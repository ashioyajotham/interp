class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, output):
        loss = self.loss_function(output, y)
        grad = self.loss_derivative(output, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def loss_function(self, output, y):
        # Implement loss function (e.g., cross-entropy)
        pass

    def loss_derivative(self, output, y):
        # Implement derivative of loss function
        pass

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            # Update weights here
            pass

    def predict(self, X):
        return self.forward(X)