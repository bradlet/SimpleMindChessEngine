# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np

ETA = 0.1
BIAS = 1
HIDDEN_LAYER_UNITS = 64
INPUTS = 768
OUTPUTS = 2


# Multi-Layered Neural Network implementation
# One input layer, Two Same-Size Hidden Layers, One Output Layer.
#   | Intended inputs == flattened bitmap representation of a chess board (vector of length 768)
class BoardStateLearner:

    def __init__(self, dataset):
        self.data = dataset
        self.input_to_first_hidden_weights = np.random.rand(INPUTS, HIDDEN_LAYER_UNITS)
        self.first_to_second_hidden_weights = np.random.rand(HIDDEN_LAYER_UNITS, HIDDEN_LAYER_UNITS)
        self.second_hidden_to_output_weights = np.random.rand(HIDDEN_LAYER_UNITS, 2)
        # REMEMBER TO USE SOFTMAX TO TURN OUTPUT INTO PROBABILITIES

    # Using Sigmoid as the activation function.
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def output_MSE(true_label, activation_vector):
        error_vector = true_label - activation_vector
        return sum(np.square(error_vector)) / len(error_vector)

    # Include sigmoid squashing
    def activations(self, data_vector, weights):
        return self.sigmoid(np.dot(data_vector, np.transpose(weights)))


