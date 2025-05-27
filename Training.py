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
    def __init__(self, episodes_nb= 100, neural_network_list= [24, 24, 24], learning_rate= 0.01):
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
            self.evaluate(episode)
        final_activation = self.forward_propagartion()
        gradients_dict = self.backward_propagation(final_activation)
        self.update_gradients(gradients_dict)
        self.evaluate()
    
    def initialization(self):
        """
        Normalize data.
        Initialiser les poids/biais pour qu'ils correspondent au réseau de neurones.
        poids = liste avec chaque element correspondant a chaque couche. chaque element est tableau [nb neurones c, nb neurones c-1(=entrée de c)]
        biais = liste avec element pour chaque couche. chaque element est tableau [nb neurones c, 1]
        Ajouter neurone de sortie.
        """
        new_array = np.zeros((self.data_results.shape[0], 2), dtype=np.float64)
        new_array[self.data_results == 'M', 1] = 1
        new_array[self.data_results == 'B', 0] = 1
        self.data_results = new_array
        

        #normalize data
        self.data_measurements = (self.data_measurements - np.mean(self.data_measurements, axis=0)) / (np.std(self.data_measurements, axis=0) + 1e-8)
        
        new_array = np.zeros((self.evaluation_data_results.shape[0], 2), dtype=np.float64)
        new_array[self.evaluation_data_results == 'M', 1] = 1
        new_array[self.evaluation_data_results == 'B', 0] = 1
        self.evaluation_data_results = new_array
        
        #normalize data
        self.evaluation_data_measurements = (self.evaluation_data_measurements - np.mean(self.evaluation_data_measurements, axis=0)) / (np.std(self.evaluation_data_measurements, axis=0) + 1e-8)
        
        input_layer_size = self.data_measurements.shape[1]
        self.nn_list = [input_layer_size] + self.nn_list + [2]
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
        activation.append(self.softmax(z_layer))
        return activation


    def backward_propagation(self, activation):
        """
        Calculates the gradients of each neuron for each layer.
        Those gradients will be used to update the weights and biases of each neuron in each layer.
        """
        dw = []
        db = []
        y_true = self.data_results.T
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

    def softmax(self, z):
        """
        Softmax is used for the output layer.
        """
        z = np.clip(z, -500, 500)
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def relu(self, z):
        """
        relu formula is used for every layer except the output layer.
        """
        return np.maximum(0, z)
    
    def log_loss(self, predictions, y_true):
        y_true = y_true.T
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(predictions))
        return loss
    
    def accuracy(self, predictions, true_results):
        y_true = true_results.T
        predicted_classes = np.argmax(predictions, axis=0)
        true_classes = np.argmax(y_true, axis=0)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy
    
    def evaluate(self, epoch=-1):
        """
        Evaluates the model using the evaluation data set to test the model on data it's not trained on.
        This does not update the model's weights/biases.
        """
        if epoch ==-1:
            epoch = self.epoch
        eval_activation = self.eval_forward_prop(self.evaluation_data_measurements.T)
        train_activation = self.eval_forward_prop(self.data_measurements.T)
        print(f"Epoch: {epoch}/{self.epoch}   Evaluation accuracy: {self.accuracy(eval_activation, self.evaluation_data_results)}   Evaluation loss: {self.log_loss(eval_activation, self.evaluation_data_results)}   Training accuracy {self.accuracy(train_activation, self.data_results)}   Training loss: {self.log_loss(train_activation, self.data_results)}")

    def eval_forward_prop(self, data):
        activation = data
        for layer in range(len(self.nn_list) - 2):
            z = self.weights[layer].dot(activation) + self.biases[layer]
            activation = self.relu(z)
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z = self.weights[last_layer].dot(activation) + self.biases[last_layer]
        activation = self.softmax(z)
        return activation

t = Training()
t.train()
