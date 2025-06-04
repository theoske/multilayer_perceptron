import numpy as np
import pickle

class Predicting:
    """
    This class loads a trained model and uses it to make predictions on the data of a csv file.
    """
    def __init__(self, model_filename = "model", data_to_predict="validation_data.csv"):
        self.data_measurements = np.genfromtxt(data_to_predict,delimiter=",",dtype=np.float64)[:,1:]
        self.data_results = np.genfromtxt(data_to_predict,delimiter=",",dtype=np.str_)[:,0]
        self.model_filename = model_filename
        self.weights = []
        self.biases = []
        self.topology = []
    
    def predict(self):
        """
        Uses the loaded model to make the predictions by doing a forward propagation.
        It does not update the model's parameters.
        """
        self.initialization()
        activation = self.forward_propagation()
        last_layer_activation = activation[len(self.topology) - 1]
        predictions = self.get_one_dim_pred(last_layer_activation)
        loss = self.binary_cross_entropy_loss(predictions)
        accu = self.accuracy(predictions)
        print(f"{self.model_filename} loss: {round(loss, 2)} accuracy: {round(accu, 2)}")
    
    def initialization(self):
        """
        Initialize data_results in a 1D array to be able to do the binary cross-entropy loss
        instead of the categorical one like in the training program.
        It also normalizes the data measurements.
        """
        self.data_results[self.data_results == 'M'] = 1
        self.data_results[self.data_results == 'B'] = 0
        self.data_results = self.data_results.astype(np.float64)

        self.data_measurements = (self.data_measurements - np.mean(self.data_measurements, axis=0)) / (np.std(self.data_measurements, axis=0) + 1e-8)

        self.load_model()
    
    def forward_propagation(self):
        """
        This forward propagation is the same as the one in training.
        It uses the softmax activation function for the output layer
        and relu for the others.
        """
        activation = [self.data_measurements.T]
        for layer in range(len(self.topology) - 2):
            z_layer = self.weights[layer].dot(activation[layer]) + self.biases[layer]
            activation.append(self.relu(z_layer))
        last_layer = len(self.topology) - 2
        z_layer = self.weights[last_layer].dot(activation[last_layer]) + self.biases[last_layer]
        activation.append(self.softmax(z_layer))
        return activation

    def relu(self, z):
        """
        relu formula is used for every layer except the output layer.
        It prevents the vanishing gradients problem which causes
        the weights to become extremely small later during the backpropagation.
        """
        return np.maximum(0, z)
    
    def softmax(self, z):
        """
        Softmax is used for the output layer.
        """
        z = np.clip(z, -500, 500)
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def binary_cross_entropy_loss(self, predictions):
        """
        Loss function for binary problems.
        Quantifies the difference between the predictions and the
        real values.
        """
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        m = predictions.shape[0]
        one_array = np.full(predictions.shape, 1)
        l = -(1/m) * np.sum(self.data_results * np.log(predictions) + (one_array-self.data_results) * np.log(1-predictions))
        return l
    
    def accuracy(self, predictions):
        """
        Returns the percentage of right answers.
        """
        predictions = np.where(predictions > 0.5, 1, 0)
        result_array = np.where(predictions == self.data_results, True, False)
        r = np.count_nonzero(result_array)
        return r / result_array.shape[0]
    
    def get_one_dim_pred(self, predictions):
        """
        Predictions is (2, N) array. This method returns predictions in a (1, N) array.
        """
        predictions = predictions.astype(np.float64)
        one_d_pred = predictions[1, :]
        return one_d_pred
    
    def load_model(self):
        """
        Loads the model (weights/biases/network topology)
        using the pickle library.
        """
        with open("models/" + self.model_filename, 'rb') as f:
            imported_model = pickle.load(f)
        self.weights = imported_model["weights"]
        self.biases = imported_model["biases"]
        self.topology = imported_model["topology"]
