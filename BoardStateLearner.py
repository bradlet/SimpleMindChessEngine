# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
import chess
from ChessHelper import ChessHelper as ch

from keras.models import Sequential
from keras.layers import Dense

MODEL_SAVE_FILE_WITHOUT_TYPE = "data/model"
INPUT_UNITS = 768
HIDDEN_LAYER_UNITS = 64
OUTPUT_UNITS = 2


# Multi-Layered Neural Network implementation
# One input layer, Two Same-Size Hidden Layers, One Output Layer.
#   | Intended inputs == flattened bitmap representation of a chess board (vector of length 768)
class BoardStateLearner:

    def __init__(self, dataset):
        if dataset != "EMPTY":
            # Transpose transpose to select by column. Could probably do this in more cleanly, but numpy hsplit() makes
            # an array of arrays, and I figured any cleaning that up would be just as messy.
            data = np.transpose(np.transpose(dataset)[:-1])
            data_labels = ch.label_one_hot_encode(np.transpose(np.transpose(dataset)[-1]))

            # Delineate training and test data
            partition_index = int(len(data) * .8)
            self.training_data = data[:partition_index]
            self.test_data = data[partition_index:]
            self.training_labels = data_labels[:partition_index]
            self.test_labels = data_labels[partition_index:]

            # Setup model w/ Keras
            self._model = Sequential()
            self._model.add(Dense(HIDDEN_LAYER_UNITS, input_dim=INPUT_UNITS, activation='relu'))
            self._model.add(Dense(HIDDEN_LAYER_UNITS, activation='relu'))
            self._model.add(Dense(OUTPUT_UNITS, activation='softmax'))
            self._model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])

    def train(self, epochs):
        return self._model.fit(self.training_data, self.training_labels,
                               validation_data=(self.test_data, self.test_labels), epochs=epochs)

    def save_model(self):
        model_json = self._model.to_json()
        # Save model info to json file
        with open(MODEL_SAVE_FILE_WITHOUT_TYPE + ".json", "w") as json_file:
            json_file.write(model_json)
        # Save model weights in separate data file
        self._model.save_weights(MODEL_SAVE_FILE_WITHOUT_TYPE + ".h5")

    def overwrite_model(self, new_model):
        self._model = new_model

    def best_next_move(self, current_board_fen, current_turn):
        board = chess.Board()
        board.set_fen(current_board_fen)  # Sets board to given state represented by the fen notation input
        board.turn = current_turn
        next_move_board_bitmaps = ch.legal_moves_bitmaps(board)

        if len(next_move_board_bitmaps) <= 1:
            # Need to make instance type 2d array for keras model.predict
            next_move_board_bitmaps = np.expand_dims(current_board_fen, axis=0)

        predictions = np.transpose(self._model.predict(next_move_board_bitmaps))
        # We are going to maximise on win probability for who's turn it is.
        predictions = predictions[1] if current_turn == chess.WHITE else predictions[0]

        return list(board.legal_moves)[int(np.argmax(predictions))]
