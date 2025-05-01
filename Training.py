import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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
        self.training_real_values = np.genfromtxt("training_data.csv",delimiter=",", dtype=str, usecols=0)
        self.weights = np.zeros(self.training_data_array.shape[1])
        self.bias = 0.0
        self.learning_rate = 0.001

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
        self.training_real_values[self.training_real_values == 'B'] = 0
        self.training_real_values[self.training_real_values == 'M'] = 1
        self.training_real_values = self.training_real_values.astype(np.float64)
        #print(self.training_real_values)
        #print(self.weights.shape)
        #print(self.training_data_array.shape)
        l = []
        for episode in range(10000):
            a = self.model()            
            l.append(self.log_loss(a))
            self.gradient_descent(a)
        y_pred = self.predict()
        #print(y_pred)
        print(accuracy_score(self.training_real_values, y_pred))
        #plt.plot(l)
        #plt.show()
    
    def predict(self):
        A = self.model()
        return (A>=0.5)
    
    def model(self):
        z = np.dot(self.training_data_array, self.weights) + self.bias
        z = np.clip(z, -500, 500)
        a = self.sigmoid(z)
        return a

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def log_loss(self, A):
        epsilon = 1e-15
        A = np.clip(A, a_min= epsilon, a_max = 1-epsilon)
        y = self.training_real_values
        m = len(y)
        l = -(1/m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        return l
    
    def gradient_descent(self, model_predictions):
        m = len(model_predictions)
        self.weights = self.weights - self.learning_rate * ((1 / m) * (np.dot(self.training_data_array.T, (model_predictions - self.training_real_values))))
        self.bias = self.bias - self.learning_rate * ((1 / m) * np.sum(model_predictions - self.training_real_values))
        

t = Training()
t.training()