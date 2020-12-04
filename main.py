# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
import chess.svg
from BoardStateLearner import BoardStateLearner as BSL

from DatasetBuilder import DATASET_FILE_PATH


def load_data(name):
    return np.loadtxt(fname=name, delimiter=" ", dtype=np.dtype(np.dtype("float32")))


if __name__ == '__main__':
    board = chess.Board()
    # print(board.legal_moves)

    dataset = load_data(DATASET_FILE_PATH)
    learner = BSL(dataset)

    history = learner.train()
    print(history)
