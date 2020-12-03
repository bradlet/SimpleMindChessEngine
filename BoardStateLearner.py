# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
from ChessHelper import ChessHelper as ch

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
        self.data = np.transpose(np.transpose(dataset)[:-1])
        self.data_labels = ch.label_one_hot_encode(np.transpose(np.transpose(dataset)[-1]))
