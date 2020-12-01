# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
import chess.svg
from ChessHelper import ChessHelper as ch
from BoardStateLearner import BoardStateLearner as BSL

from DatasetBuilder import DATASET_FILE_PATH


def load_data(name):
    return np.loadtxt(fname=name, delimiter=" ", dtype=np.dtype(type(int)))


if __name__ == '__main__':
    board = chess.Board()
    # print(board.legal_moves)

    dataset = load_data(DATASET_FILE_PATH)
    print(np.shape(dataset[0]))

