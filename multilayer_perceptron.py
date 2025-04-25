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
    def __innit__(self):
        self.training_data_array = np.genfromtxt("training_data.csv",delimiter=",", dtype=np.float64)
        self.validation_data_array = np.genfromtxt("validation_data.csv", delimiter=",",dtype=np.float64)
    
    def training(self):
        """
        30 data variables.
        1) Make a prediction with :
            Z = input_array * weights_array + bias
            A = sigmoid(Z)
        2) Calculate the error with log loss.
        3) Adjust the weights W and the bias B with the gradient descent formulas.
        4) Repeat.
        """
        
