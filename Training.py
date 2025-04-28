import numpy as np

"""
Perceptron uses 3 formulas :
 - Linear function
 - Activation function (sigmoid)
 - Log loss function.
 - Gradient descent.
And gradient descent algorithm.

The data and the formulas are vectorized using numpy.
"""

class Training():
    """
        Slicing : array[row_start:row_stop:row_step, col_start:col_stop:col_step]
    """
    def __innit__(self):
        self.training_data_array = np.genfromtxt("training_data.csv",delimiter=",",dtype=np.float64)[:,1:]
        self.validation_data_array = np.genfromtxt("validation_data.csv", delimiter=",",dtype=np.float64)[:, 1:]
        self.training_result = np.genfromtxt("training_data.csv",delimiter=",")[:, 0]
        self.validating_result = np.genfromtxt("validation_data.csv", delimiter=",")[:, 0]
        self.weights = np.zeros(30)
        self.bias = np.zeros(1)

    def training(self):
        """
        30 data variables.
        1) Make a prediction with :
            Z = input_array * weights_array + bias
            A = sigmoid(Z)
        2) Calculate the error with log loss.
        3) Adjust the weights W and the bias B with the gradient descent formulas.
        4) Repeat.
        This is the basic version without backward propagation and hidden layers.
        
        result is 0 for B and 1 for M
        """
        self.training_result = np.where(self.training_result == 'B', 0, 1)
        for episode in range(1000):
            z = self.weights @ self.training_data_array.T + self.bias[:, np.newaxis]
            a = self.sigmoid(z)
            l = self.log_loss(a)
            self.gradient_descent(l)
        
    def sigmoid(self, z):
        return (1 / (np.exp(-z) + 1))

    def log_loss(self, z):
        pass
    
    def gradient_descent(self, l):
        pass
        