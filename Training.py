import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
from math import exp, sqrt


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
    def __init__(self, neuron_per_layer_list = [24, 24, 24], episodes = 1000, learning_rate = 0.001):
        self.training_data_array = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.training_real_values = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.parameters = {}
        self.neuron_per_layer_list = neuron_per_layer_list + [1]
        self.initialization()
        self.learning_rate = learning_rate
        self.episodes = episodes
        
    def initialization(self):
        """
        W.shape = [input nb, output nb]
        b.shape = [output nb, 1]
        For W0 the input is the number of features in the training data.
        For the other W the input is the number of neuron in the current layer.
        """
        self.parameters = {}
        np.random.seed(1)
        param_dim = [self.training_data_array.shape[1]] + self.neuron_per_layer_list
        for layer in range(1, len(param_dim)):
            self.parameters["W" + str(layer)] = np.full((param_dim[layer], param_dim[layer - 1]), sqrt(6/54))
            self.parameters["b" + str(layer)] = np.full((param_dim[layer], 1), sqrt(6/54))
        for k, v in self.parameters.items():
            print(k, v)

    def training(self):
        """
        30 data variables.
        1) Make a prediction with (= forward_propagation):
            Z = input_array * weights_array + bias
            A = sigmoid(Z)
        2) Calculate the error with log loss.
        3) Adjust the weights W and the bias B with the gradient descent formulas (= back_propagation).
        4) Repeat.
        This is the basic version without backward propagation and hidden layers.
        
        result is 0 for B and 1 for M
        
        The weights and bias are saved in a modelname.json file.
        
        Uses X in the first layer and A-1 in the next ones.
        """
        self.training_real_values[self.training_real_values == 'B'] = 0
        self.training_real_values[self.training_real_values == 'M'] = 1
        self.training_real_values = self.training_real_values.astype(np.float64).reshape(-1, 1)
        l = []
        for episode in range(self.episodes):
            a = self.forward_propagation()            
            l.append(self.log_loss(a))
            gradients = self.back_propagation(a)
            self.parameters_update(gradients)
        self.save_model()
        print(accuracy_score(self.training_real_values.flatten(), self.predict().flatten()))
        plt.plot(l)
        #plt.show()
    
    def predict(self): # mettre dans programme de predictions.
        a = self.forward_propagation()
        predictions = (a["A" + str(len(self.neuron_per_layer_list))] >= 0.5).astype(int)
        #print(a)
        return predictions.T
    
    def forward_propagation(self):
        layers_nb = len(self.neuron_per_layer_list)
        a = {"A0" : self.training_data_array.T}
        for layer in range(1, layers_nb + 1):
            z = self.parameters['W' + str(layer)].dot(a['A' + str(layer - 1)]) + self.parameters['b' + str(layer)]
            a["A" + str(layer)] = self.softmax(z) if (layer == layers_nb) else self.sigmoid(z)
            a["Z" + str(layer)] = z
        return a

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))
    
    def softmax(self, z):
        return np.exp(z) / (exp(1) + exp(0))

    def log_loss(self, a):
        """
        Takes the activation of the output layer and calculate the
        error with the log loss/binary cross entropy function.
        """
        a = a["A" + str(len(self.neuron_per_layer_list))]
        epsilon = 1e-15
        a = np.clip(a, a_min= epsilon, a_max = 1-epsilon)
        y = self.training_real_values.T
        m = y.shape[1]
        l = -(1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return l
    
    def back_propagation(self, model_predictions):
        layer_count = len(self.neuron_per_layer_list)
        a_final = model_predictions["A" + str(layer_count)]
        y = self.training_real_values.T
        dZ = a_final - y
        m = self.training_real_values.shape[0]
        gradients = {}
        for layer in reversed(range(1, layer_count + 1)):
            a_prev = model_predictions["A" + str(layer - 1)]
            gradients["dW" + str(layer)] = 1 / m * np.dot(dZ, a_prev.T)
            gradients["db" + str(layer)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if layer > 1:
                dZ = np.dot(self.parameters["W" + str(layer)].T, dZ)  * model_predictions['A' + str(layer - 1)] * (1 - model_predictions['A' + str(layer - 1)])
        return gradients
    
    def parameters_update(self, gradients):
        layer_count = len(self.neuron_per_layer_list)
        for layer in range(1, layer_count + 1):
            self.parameters["W" + str(layer)] -= self.learning_rate * gradients["dW" + str(layer)]
            self.parameters["b" + str(layer)] -= self.learning_rate * gradients["db" + str(layer)]
    
    def save_model(self):
        model_dict = {
            "parameters"    : self.parameters,
            "topology"      : self.neuron_per_layer_list
        }
        with open('model', 'wb+') as f:
            pickle.dump(model_dict, f)
        

t = Training(episodes=10000, learning_rate=0.0001)
t.training()