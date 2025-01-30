# AI Intepretability
## Overview
This project implements a neural network from scratch using NumPy. It is designed to help understand the inner workings of neural networks, including forward propagation, backpropagation, and various layer functionalities.

## Project Structure
```
neural-net-numpy
├── src
│   ├── network.py         # Defines the NeuralNetwork class
│   ├── layers.py          # Contains layer classes like Dense and Activation
│   ├── utils
│   │   ├── data_loader.py  # Functions for loading and preprocessing MNIST
│   │   └── metrics.py      # Functions for calculating performance metrics
│   └── train.py           # Main training loop for the neural network
├── tests
│   └── test_network.py     # Unit tests for the NeuralNetwork class
├── requirements.txt        # Lists project dependencies
├── README.md               # Project documentation
└── .gitignore              # Specifies files to ignore in Git
```

## Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd neural-net-numpy
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To train the neural network on the MNIST dataset, run the following command:
```
python src/train.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.