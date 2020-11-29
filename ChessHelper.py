# Simple Mind Chess Engine
# Bradley Thompson

import numpy as np
from chess import WHITE, BLACK


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

    # Returns a numpy array of shape [64, 12] (squares on chess board, number of possible pieces including both sides).
    # One row from this array will be all zeroes if a square is empty, or it will have 1 or -1 in the index
    # corresponding to whatever piece is in that square.
    # | I did not come up with this idea for data representation, idea from:
    # | https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
    @staticmethod
    def bitmap_representation(board):
        # representation is dependent on which side's turn it is.
        player, opponent = (1, -1) if board.turn is WHITE else (-1, 1)
        array = np.zeros((64, 12), dtype=type(int))
        array_ctr = 0

        rows = board.board_fen().split('/')
        for row in rows:
            for char in row:
                # In fen notation, a number denotes how many concurrent empty squares exist at that point in a row.
                if not char.isalpha():
                    array_ctr += int(char)
                else:
                    array[array_ctr][ChessHelper.__pieceMap[char]] = player if char.islower() else opponent
                    array_ctr += 1

        return array





