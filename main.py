# Simple Mind Chess Engine
# Bradley Thompson

import chess.svg
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from BoardStateLearner import BoardStateLearner as BSL
from BoardStateLearner import MODEL_SAVE_FILE_WITHOUT_TYPE
from DatasetBuilder import DATASET_FILE_PATH


# Pyplot layout taken from https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
# I could have changed it arbitrarily but I liked the way the one in this tutorial looked, and this isn't really
# directly relevant to the machine learning part of the project. So I figured it was fine to copy!
def plot_accuracy(history):
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# Not trying to really follow a scalable pattern, just doing easy stuff to make this play project work.
# This is the entry point for the REST service that uses this app.
def best_move_call(side, fen):
    json_file = open(MODEL_SAVE_FILE_WITHOUT_TYPE+".json", "r")
    model_json = json_file.read()
    json_file.close()

    trained_model = model_from_json(model_json)
    trained_model.load_weights(MODEL_SAVE_FILE_WITHOUT_TYPE+".h5")

    neural_net = BSL("EMPTY")
    neural_net.overwrite_model(trained_model)
    return str(neural_net.best_next_move(fen, side))


def load_data(name):
    return np.loadtxt(fname=name, delimiter=" ", dtype=np.dtype(np.dtype("float32")))


if __name__ == '__main__':
    dataset = load_data(DATASET_FILE_PATH)
    learner = BSL(dataset)

    board = chess.Board()
    training_history = learner.train(epochs=25)

    learner.save_model()

    plot_accuracy(training_history)
    plot_loss(training_history)
