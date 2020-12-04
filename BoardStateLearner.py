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

        # Delineate into training and test, then use validation_data=(testdata, testdatalabels) in model.fit

        self._model = Sequential()
        self._model.add(Dense(HIDDEN_LAYER_UNITS, input_dim=INPUT_UNITS, activation='relu'))
        self._model.add(Dense(HIDDEN_LAYER_UNITS, activation='relu'))
        self._model.add(Dense(OUTPUT_UNITS, activation='softmax'))

        self._model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])

    def train(self, epochs):
        return self._model.fit(self.data, self.data_labels, epochs=epochs)

    def eval(self):
        test = self.data[8:10]
        return self._model.predict(test)
