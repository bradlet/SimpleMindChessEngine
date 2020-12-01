# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
import chess.svg
from ChessHelper import ChessHelper as ch
from BoardStateLearner import BoardStateLearner as BSL

from DatasetBuilder import DATASET_FILE_PATH

# NEED TO PRE PROCESS A BIT MAYBE? OR JUST USE ARRAY STRIPPING TO GET CLASS LABELS / NOT USE THEM


def load_data(name):
    return np.loadtxt(fname=name, delimiter=" ", dtype=np.dtype(type(int)))


if __name__ == '__main__':
    board = chess.Board()
    # print(board.legal_moves)

    dataset = load_data(DATASET_FILE_PATH)
    learner = BSL(dataset)

    learner.calculate_MSE([1], [2, 3, 4])
