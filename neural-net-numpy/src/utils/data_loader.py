def load_mnist_data():
    import numpy as np
    from tensorflow.keras.datasets import mnist

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return (x_train, y_train), (x_test, y_test)

def split_data(x, y, split_ratio=0.8):
    split_index = int(len(x) * split_ratio)
    return (x[:split_index], y[:split_index]), (x[split_index:], y[split_index:])