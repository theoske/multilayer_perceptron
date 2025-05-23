import numpy as np
import pickle

class Training():
    """
    entrées:
        - nb episodes
        - reseau de neurone
        - learning_rate
    variables de classe:
        - données d'entrainements.
        - forme réseau de neurones.
        - learning rate.
        - poids biais.
        - nb episodes.
    """
    def __init__(self, episodes_nb= 10000, neural_network_list= [24, 24, 24], learning_rate= 0.001):
        self.epoch = episodes_nb
        self.nn_list = neural_network_list
        self.learning_rate = learning_rate
        self.data_measurements = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.data_results = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.evaluation_data_measurements = np.genfromtxt("validation_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.evaluation_data_results = np.genfromtxt("validation_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.weights = []
        self.biases = []
    
    def train(self):
        self.initialization()
        for episode in range(self.epoch):
            activation = self.forward_propagartion()
            gradients_dict = self.backward_propagation(activation)
            self.update_gradients(gradients_dict)
        final_activation = self.forward_propagartion()
        final_loss = self.log_loss(final_activation[-1])
        final_accuracy = self.accuracy(final_activation[-1], self.data_results)
        print(f"Final - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        self.evaluate()
    
    def initialization(self):
        """
        Normalize data.
        Initialiser les poids/biais pour qu'ils correspondent au réseau de neurones.
        poids = liste avec chaque element correspondant a chaque couche. chaque element est tableau [nb neurones c, nb neurones c-1(=entrée de c)]
        biais = liste avec element pour chaque couche. chaque element est tableau [nb neurones c, 1]
        Ajouter neurone de sortie.
        """
        self.data_results[self.data_results == 'B'] = 0
        self.data_results[self.data_results == 'M'] = 1
        self.data_results = self.data_results.astype(np.float64)
        
        #normalize data
        self.data_measurements = (self.data_measurements - np.mean(self.data_measurements, axis=0)) / (np.std(self.data_measurements, axis=0) + 1e-8)
        
        self.evaluation_data_results[self.evaluation_data_results == 'B'] = 0
        self.evaluation_data_results[self.evaluation_data_results == 'M'] = 1
        self.evaluation_data_results = self.evaluation_data_results.astype(np.float64)
        
        #normalize data
        self.evaluation_data_measurements = (self.evaluation_data_measurements - np.mean(self.evaluation_data_measurements, axis=0)) / (np.std(self.evaluation_data_measurements, axis=0) + 1e-8)
        
        input_layer_size = self.data_measurements.shape[1]
        self.nn_list = [input_layer_size] + self.nn_list + [1]
        np.random.seed(2)
        for layer in range(1, len(self.nn_list)):
            self.weights.append(np.random.rand(self.nn_list[layer], self.nn_list[layer-1]))
            self.biases.append(np.random.rand(self.nn_list[layer], 1))

    def forward_propagartion(self):
        """
        Calculates the activation of each layer with the weights and biases
        and stores it for the backward propagation.
        """
        z = []
        activation = [self.data_measurements.T]
        #uses relu function for hidden layers.
        for layer in range(len(self.nn_list) - 2):# -2 car compte pas la couche dentree ni la couche de sortie (l'index qui commence a 0 est prit en compte par in qui va jusqua len exclue)
            z_layer = self.weights[layer].dot(activation[layer]) + self.biases[layer]
            z.append(z_layer)
            activation.append(self.relu(z_layer))
        #uses softmax function to calculate output.
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z_layer = self.weights[last_layer].dot(activation[last_layer]) + self.biases[last_layer]
        z.append(z_layer)
        activation.append(self.sigmoid(z_layer))
        return activation


    def backward_propagation(self, activation):
        """
        Calculates the gradients of each neuron for each layer.
        Those gradients will be used to update the weights and biases of each neuron in each layer.
        """
        dw = []
        db = []
        y_true = self.data_results.reshape(1, -1)
        dz = activation[-1] - y_true
        layers = len(self.nn_list) - 1
        m = y_true.shape[1]
        for layer in reversed(range(layers)):
            dw.insert(0, (1/m) * np.dot(dz, activation[layer].T))
            db.insert(0, (1/m) * np.sum(dz, axis=1, keepdims=True))
            if layer > 0: # not necessary but prevent unnecessary calculations
                # For ReLU derivative
                relu_derivative = (activation[layer] > 0).astype(float)
                dz = np.dot(self.weights[layer].T, dz) * relu_derivative
        return {"dw":dw, "db":db}
    
    def update_gradients(self, gradients_dict):
        dw = gradients_dict["dw"]
        db = gradients_dict["db"]
        layers = len(self.nn_list) - 1
        for layer in range(layers):
            self.weights[layer] = self.weights[layer] - (self.learning_rate * dw[layer])
            self.biases[layer] = self.biases[layer] - (self.learning_rate * db[layer])

    def sigmoid(self, z):
        """
        Softmax is used for the output layer.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        """
        relu formula is used for every layer except the output layer.
        """
        #z = np.clip(z, -200, 200)
        return np.maximum(0, z)
    
    def log_loss(self, predictions):
        y_true = self.data_results.reshape(1, -1)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(predictions) + (1 - y_true) * np.log(1 - predictions))
        return loss
    
    def accuracy(self, predictions, true_results):
        y_true = true_results.reshape(1, -1)
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y_true)
        return accuracy
    
    def evaluate(self):
        """
        Evaluates the model using the evaluation data set to test the model on data it's not trained on.
        This does not update the model's weights/biases.
        """
        activation = self.evaluation_data_measurements.T
        for layer in range(len(self.nn_list) - 2):
            z = self.weights[layer].dot(activation) + self.biases[layer]
            activation = self.relu(z)
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z = self.weights[last_layer].dot(activation) + self.biases[last_layer]
        activation = self.sigmoid(z)
        print(f"Evaluation accuracy: {self.accuracy(activation, self.evaluation_data_results)}")

t = Training()
t.train()
