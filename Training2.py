import numpy as np
import pickle
from math import exp, sqrt

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
    def __init__(self, episodes_nb= 10000, neural_network_list= [24, 24, 24], learning_rate= 0.0005):
        self.episodes_nb = episodes_nb
        self.nn_list = neural_network_list
        self.learning_rate = learning_rate
        self.data_measurements = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.data_results = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.weights = []
        self.biases = []
    
    def train(self):
        self.initialization()
        activation = self.forward_propagartion()
        gradients = self.backward_propagation(activation)
    
    def initialization(self):
        """
        Initialiser les poids/biais pour qu'ils correspondent au réseau de neurones.
        poids = liste avec chaque element correspondant a chaque couche. chaque element est tableau [nb neurones c, nb neurones c-1(=entrée de c)]
        biais = liste avec element pour chaque couche. chaque element est tableau [nb neurones c, 1]
        Ajouter neurone de sortie.
        """
        input_layer_size = self.data_measurements.shape[1]
        self.nn_list = [input_layer_size] + self.nn_list + [1]
        print(f"Topology : {self.nn_list}")
        np.random.seed(2)
        for layer in range(1, len(self.nn_list)):
            self.weights.append(np.random.rand(self.nn_list[layer], self.nn_list[layer-1]))
            self.biases.append(np.random.rand(self.nn_list[layer], 1))
            #self.weights.append(np.full((self.nn_list[layer], self.nn_list[layer-1]), sqrt(6/31)))
            #self.biases.append(np.full((self.nn_list[layer], 1), sqrt(6/31)))

        for value in self.weights:
            print(f"weight shape{value.shape}")
        for v in self.biases:
            print(f"biases shape{v.shape}")

    def forward_propagartion(self):
        """
        Calculates the activation of each layer with the weights and biases
        and stores it for the backward propagation.
        """
        z = []
        activation = [self.data_measurements.T]
        #uses sigmoid function for hidden layers.
        for layer in range(len(self.nn_list) - 2):# -2 car compte pas la couche dentree ni la couche de sortie (l'index qui commence a 0 est prit en compte par in qui va jusqua len exclue)
            print(f"layer: {layer}  weight shape: {self.weights[layer].shape}")
            z.append(self.weights[layer].dot(activation[layer]) + self.biases[layer])
            activation.append(self.sigmoid(z[layer]))
        #uses softmax function to calculate output.
        print(f"nn_listlen: {len(self.nn_list)}, activationlen: {len(activation)}, weightlen: {len(self.weights)}")
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z.append(self.weights[last_layer].dot(activation[last_layer]) + self.biases[last_layer])
        activation.append(self.sigmoid(z[last_layer]))
        for v in activation:
            print(f"activation shape{v.shape}")
        print(activation[4])
        return activation


    def backward_propagation(self, activation):
        """
        Calculates the gradients of each neuron for each layer.
        Those gradients will be used to update the weights and biases of each neuron in each layer.
        """
        pass

    def softmax(self, z):
        """
        Softmax is used for the output layer.
        """
        return np.exp(z) / (exp(1) + 1)

    def sigmoid(self, z):
        """
        Sigmoid formula is used for every layer except the output layer.
        """
        return 1 / (1 + np.exp(-z))

t = Training()
t.train()
