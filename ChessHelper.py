# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
from chess import WHITE


class ChessHelper:

    # A mapping from char to binary attribute array index -- capitalization denotes separate players
    # [pawn, rook, knight, bishop, queen, king, Pawn, Rook, Knight, Bishop, Queen, King]
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

    # Convert a board into a vector of len 768, 64 x 12
    # | 12 features for each piece (6 per color)
    # | So, for each of the 64 squares, there are 12 features.
    # -> Return: A numpy array of shape [64, 12]
    # One row from this array will be all zeroes if a square is empty, or it will have 1 or -1 in the index
    # corresponding to whatever piece is in that square.
    @staticmethod
    def bitmap_representation(board):
        # I did not come up with this idea for data representation, idea from:
        # https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
        array = np.zeros((64, 12), dtype=type(int))
        array_ctr = 0

        rows = board.board_fen().split('/')
        for row in rows:
            for char in row:
                # In fen notation, a number denotes how many concurrent empty squares exist at that point in a row.
                if not char.isalpha():
                    array_ctr += int(char)
                else:
                    # -1 DENOTES BLACK, 1 DENOTES WHITE
                    array[array_ctr][ChessHelper.__pieceMap[char]] = -1 if char.isupper() else 1
                    array_ctr += 1

        return array

    # Converts class data labels into a binary one hot encoding representation
    @staticmethod
    def label_one_hot_encode(data_labels):
        # np.array doesn't know how to handle maps apparently, so need the list conversion as an intermediary step.
        return np.array(list(map(lambda x: [1, 0] if x == -1 else [0, 1], data_labels)))



