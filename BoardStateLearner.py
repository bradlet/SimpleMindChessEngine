# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
from ChessHelper import ChessHelper as ch

from keras.models import Sequential
from keras.layers import Dense

INPUT_UNITS = 768
HIDDEN_LAYER_UNITS = 64
OUTPUT_UNITS = 2


# Multi-Layered Neural Network implementation
# One input layer, Two Same-Size Hidden Layers, One Output Layer.
#   | Intended inputs == flattened bitmap representation of a chess board (vector of length 768)
class BoardStateLearner:

    def __init__(self, dataset):
        self.data = np.transpose(np.transpose(dataset)[:-1])
        self.data_labels = ch.label_one_hot_encode(np.transpose(np.transpose(dataset)[-1]))

        self._model = Sequential()
        self._model.add(Dense(HIDDEN_LAYER_UNITS, input_dim=INPUT_UNITS, activation='relu'))
        self._model.add(Dense(HIDDEN_LAYER_UNITS, activation='relu'))
        self._model.add(Dense(OUTPUT_UNITS, activation='softmax'))

        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    def train(self):
        return self._model.fit(self.data, self.data_labels, epochs=10)
