from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from unittest.mock import patch

class Evaluation:
    def __init__(self, model_filename = "model.json"):
        self.validation_data_array = np.genfromtxt("validation_data.csv",delimiter=",",dtype=np.float64)[:,1:]
        self.validation_result = np.genfromtxt("validation_data.csv",delimiter=",", dtype=str, usecols=0)
        self.model_filename = model_filename
        self.parameters = {}
        self.neuron_per_layer_list = []
    
    def evaluation(self):
        self.load_model()
        self.validation_result[self.validation_result == 'B'] = 0
        self.validation_result[self.validation_result == 'M'] = 1
        self.validation_result = self.validation_result.astype(np.float64)
        y_prediction = self.predict()
        #print(y_prediction)
        #print(self.validation_result)
        print(accuracy_score(self.validation_result, y_prediction))
    
    def predict(self):
        A = self.forward_propagation()
        return A >= 0.5
    
    def forward_propagation(self):
        layers_nb = len(self.neuron_per_layer_list)
        a = {"A0" : self.validation_data_array.T}
        for layer in range(1, layers_nb + 1):
            z = self.parameters['W' + str(layer)].dot(a['A' + str(layer - 1)]) + self.parameters['b' + str(layer)]
            a["A" + str(layer)] = self.sigmoid(z)
        a = a["A" + str(len(self.neuron_per_layer_list))]
        return a.T[:,:1]
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return (1 / (1 + np.exp(-z)))
    
    def load_model(self):
        with open(self.model_filename, 'rb') as f:
            import_model = pickle.load(f)
        self.parameters = import_model["parameters"]
        self.neuron_per_layer_list = import_model["topology"]

eval = Evaluation("model")
eval.evaluation()