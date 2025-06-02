import numpy as np
import pickle
import matplotlib.pyplot as plt

class Training():
    """
    This class trains, evaluates and saves a model.
    """
    def __init__(self, epochs= 100, neural_network_list= [24, 24, 24], learning_rate= 0.01, model_filename="model"):
        self.epochs = epochs
        self.nn_list = neural_network_list
        self.learning_rate = learning_rate
        self.data_measurements = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.data_results = np.genfromtxt("training_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.evaluation_data_measurements = np.genfromtxt("validation_data.csv", delimiter=",", dtype=np.float64)[:,1:]
        self.evaluation_data_results = np.genfromtxt("validation_data.csv", delimiter=",", dtype=np.str_)[:,0]
        self.weights = []
        self.biases = []
        self.learning_stats = {}
        self.model_filename = model_filename
    
    def train(self):
        """
        Creates a model with random parameters then goes through
        the training loop:
            - Makes predictions for each training data.
            - Calculates the gradients based on the predictions error.
            - Adjusts the parameters with the gradients.
            - Evaluates the model's performances.
            - Repeat.
        """
        try:
            self.initialization()
        except:
            print("ERROR : Training parameters invalid for initialization.")
            exit(-1)
        for episode in range(self.epochs):
            activation = self.forward_propagartion()
            gradients_dict = self.backward_propagation(activation)
            self.update_gradients(gradients_dict)
            self.evaluate(episode)
        final_activation = self.forward_propagartion()
        gradients_dict = self.backward_propagation(final_activation)
        self.update_gradients(gradients_dict)
        self.evaluate()
        self.save_model()
        self.show_train_stats()
    
    def initialization(self):
        """
        Normalizes training and evaluating data.
        Initializes data arrays and weights/biases
        based on the neural network topology.
        Rewrites results arrays using 2 columns, one for each
        potential outcome (B/M) and sets the elements to 0 or 1
        depending which outcome is the real one.
        """
        self.learning_stats["train_loss"] = []
        self.learning_stats["eval_loss"] = []
        self.learning_stats["train_accu"] = []
        self.learning_stats["eval_accu"] = []

        new_array = np.zeros((self.data_results.shape[0], 2), dtype=np.float64)
        new_array[self.data_results == 'M', 1] = 1
        new_array[self.data_results == 'B', 0] = 1
        self.data_results = new_array

        self.data_measurements = (self.data_measurements - np.mean(self.data_measurements, axis=0)) / (np.std(self.data_measurements, axis=0) + 1e-8)

        new_array = np.zeros((self.evaluation_data_results.shape[0], 2), dtype=np.float64)
        new_array[self.evaluation_data_results == 'M', 1] = 1
        new_array[self.evaluation_data_results == 'B', 0] = 1
        self.evaluation_data_results = new_array

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
        It uses the softmax activation function for the output layer
        and relu for the others.
        """
        activation = [self.data_measurements.T]
        for layer in range(len(self.nn_list) - 2):# -2 car compte pas la couche dentree ni la couche de sortie (l'index qui commence a 0 est prit en compte par in qui va jusqua len exclue)
            z_layer = self.weights[layer].dot(activation[layer]) + self.biases[layer]
            activation.append(self.relu(z_layer))
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z_layer = self.weights[last_layer].dot(activation[last_layer]) + self.biases[last_layer]
        activation.append(self.softmax(z_layer))
        return activation


    def backward_propagation(self, activation):
        """
        Calculates the gradients of each neuron for each layer.
        Those gradients will be used to update the weights and
        biases of each neuron in each layer.
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
        """
        Updates the weights/biases using the gradients found
        in the back propagation.
        """
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
        It's useful to avoid the vanishing gradients problem.
        """
        return np.maximum(0, z)
    
    def categorical_cross_entropy_loss(self, predictions, y_true):
        """
        Calculates the error of the output layer.
        This method is used for cases where there is an
        output for each output category.
        We use the clip method to avoid log(0) = nan.
        """
        y_true = y_true.T
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(predictions))
        return loss
    
    def accuracy(self, predictions, true_results):
        """
        Returns the percentage of right answers.
        """
        y_true = true_results.T
        predicted_classes = np.argmax(predictions, axis=0)
        true_classes = np.argmax(y_true, axis=0)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy
    
    def evaluate(self, epochs=-1):
        """
        Evaluates the model using the evaluation data set to test the model on data it's not trained on.
        This does not update the model's weights/biases.
        """
        if epochs ==-1:
            epochs = self.epochs
        eval_activation = self.eval_forward_prop(self.evaluation_data_measurements.T)
        train_activation = self.eval_forward_prop(self.data_measurements.T)
        eval_loss = self.categorical_cross_entropy_loss(eval_activation, self.evaluation_data_results)
        train_loss = self.categorical_cross_entropy_loss(train_activation, self.data_results)
        eval_accu = self.accuracy(eval_activation, self.evaluation_data_results)
        train_accu = self.accuracy(train_activation, self.data_results)
        print(f"Epoch: {epochs}/{self.epochs}   Evaluation accuracy: {round(eval_accu, 2)}   Evaluation loss: {round(eval_loss, 2)}   Training accuracy {round(train_accu, 2)}   Training loss: {round(train_loss, 2)}")
        self.learning_stats["eval_loss"].append(eval_loss)
        self.learning_stats["train_loss"].append(train_loss)
        self.learning_stats["eval_accu"].append(eval_accu)
        self.learning_stats["train_accu"].append(train_accu)
    
    def eval_forward_prop(self, data):
        """
        Forward propagation used for evaluations.
        It does not update the model's parameters.
        """
        activation = data
        for layer in range(len(self.nn_list) - 2):
            z = self.weights[layer].dot(activation) + self.biases[layer]
            activation = self.relu(z)
        last_layer = len(self.nn_list) - 2 # -2 car compte pas la couche dentree et doit prendre l'index (qui commence a 0)
        z = self.weights[last_layer].dot(activation) + self.biases[last_layer]
        activation = self.softmax(z)
        return activation
    
    def show_train_stats(self):
        """
        Uses the matplotlib library to display a
        graph that shows the evolution of the loss
        and the accuracy over training episodes.
        """
        fig, (loss, accu) = plt.subplots(1, 2)
        loss.plot(self.learning_stats["eval_loss"], color="orange")
        loss.plot(self.learning_stats["train_loss"], color="b")
        loss.set_xlabel("Epochs")
        loss.set_ylabel("Loss")
        loss.set_title("Categorical cross-entropy loss")
        loss.legend(['Evaluation', 'Training'])
        
        accu.plot(self.learning_stats["eval_accu"], color="orange")
        accu.plot(self.learning_stats["train_accu"], color="b")
        accu.set_xlabel("Epochs")
        accu.set_ylabel("Accuracy")
        accu.set_title("Learning curve")
        accu.legend(['Evaluation', 'Training'])
        
        plt.tight_layout()
        fig.set_figwidth(13)
        plt.show()
    
    def save_model(self):
        """
        Saves the model (weights/biases/network topology)
        with the pickle library.
        """
        model_dict = {"weights": self.weights, "biases": self.biases, "topology": self.nn_list}
        with open("models/" + self.model_filename, 'wb') as f:
            pickle.dump(model_dict, f)

