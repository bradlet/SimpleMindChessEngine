# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np


class ChessHelper:

    __pieceMap = {
        "p": 0,
        "r": 1,
        "n": 2,
        "b": 3,
        "q": 4,
        "k": 5,
        "P": 6,
        "R": 7,
        "N": 8,
        "B": 9,
        "Q": 10,
        "K": 11
    }

    # Returns a numpy array of shape [64, 12] (squares on chess board, number of possible pieces including both sides).
    # Total number of input units = 768.
    # I did not come up with this idea for data representation, idea from:
    # Learning to Evaluate Chess Positions with Deep Neural Networks and Limited Lookahead
    # by Matthia Sabatelli, Francesco Bidoia, Valeriu Codreanu and Marco Wiering
    # https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
    @staticmethod
    def typed_binary_representation(board):
        fen = board.board_fen()
        array = np.zeros((64, 12), dtype=type(int))
        array_ctr = 0

        rows = fen.split('/')
        for row in rows:
            for char in row:
                # In fen notation, a number denotes how many concurrent empty squares exist at that point in a row.
                if not char.isalpha():
                    array_ctr += int(char)





