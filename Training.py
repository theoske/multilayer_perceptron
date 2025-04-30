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
    def __init__(self):
        self.training_data_array = np.genfromtxt("training_data.csv",delimiter=",",dtype=np.float64)[:,1:]
        self.training_real_values = np.genfromtxt("training_data.csv",delimiter=",")[:, 0]
        self.weights = np.zeros(30)
        self.bias = 0
        self.learning_rate = 0.9995

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
        self.training_real_values = np.where(self.training_real_values == 'B', 0, 1)
        for episode in range(2):
            z = self.weights @ self.training_data_array.T + self.bias
            #print(z)
            a = self.sigmoid(z)
            l = self.log_loss(a)
            #print(l)
            self.gradient_descent(l, z)
        
    def sigmoid(self, z):
        return (1 / (np.exp(-z) + 1))

    def log_loss(self, model_predictions):
        m = self.training_data_array.shape[0]
        l = -(1/m) * np.sum(self.training_real_values * np.log(model_predictions) + (1 - self.training_real_values)* np.log(1 - model_predictions))
        return l
    
    def gradient_descent(self, model_predictions):
        m = self.training_data_array.shape[0]
        self.weights = self.weights - self.learning_rate * ((1 / m) * np.sum(self.training_data_array.T * (model_predictions - self.training_real_values)))
        self.bias = self.bias - self.learning_rate * ((1 / m) * np.sum(model_predictions - self.training_real_values))
        print(self.weights)
        print(self.bias)
        

t = Training()
t.training()