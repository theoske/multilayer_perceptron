from sklearn.metrics import accuracy_score
import numpy as np
import json

class Evaluation:
    def __init__(self, model_filename = "model.json"):
        self.validation_data_array = np.genfromtxt("validation_data.csv",delimiter=",",dtype=np.float64)[:,1:]
        self.validation_result = np.genfromtxt("validation_data.csv",delimiter=",", dtype=str, usecols=0)
        self.weights = 0
        self.bias = 0
        self.model_filename = model_filename
    
    def evaluation(self):
        self.load_model()
        self.validation_result[self.validation_result == 'B'] = 0
        self.validation_result[self.validation_result == 'M'] = 1
        self.validation_result = self.validation_result.astype(np.float64)
        y_prediction = self.predict()
        print(y_prediction)
        print(self.validation_result)
        print(accuracy_score(self.validation_result, y_prediction))
    
    def predict(self):
        A = self.model()
        return A >= 0.5
    
    def model(self):
        z = np.dot(self.validation_data_array, self.weights) + self.bias
        a = self.sigmoid(z)
        return a
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return (1 / (1 + np.exp(-z)))
    
    def load_model(self):
        with open(self.model_filename, 'r') as f:
            model_json = json.load(f)
        self.weights = np.array(model_json['weights'])
        self.bias = model_json['bias']

eval = Evaluation("model.json")
eval.evaluation()