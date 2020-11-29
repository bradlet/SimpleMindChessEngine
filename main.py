# Simple Mind Chess Engine
# Bradley Thompson

import chess
import chess.svg
from ChessHelper import ChessHelper as ch

# Convert a board into a vector of len 768, 64 x 12
#   12 features for each piece (6 per color)
#   So, for each of the 64 squares, there are 12 features.
#   [pawn, rook, knight, bishop, queen, king... opposite side]

if __name__ == '__main__':
    board = chess.Board()
    print(board.board_fen())
    ch.typed_binary_representation(board)

    # # Example of two ways to make moves
    # move = chess.Move(from_square=chess.E2, to_square=chess.E4)
    # move2 = chess.Move.from_uci("e7e5")
    # board.push(move)
    # board.push(move2)
    #
    # print(board.board_fen())

    # board.push(chess.Move.from_uci("f2f4"))
    # print(board.legal_moves)
    #
    # # So, the board will accept non-legal moves. Have to be careful about that.
    # # Board also has no concept of which turn it is as far as push_san() is concerned.
    # # board.push_san("exf4")
    # board.push_san("Ne7")
    # print(board)

