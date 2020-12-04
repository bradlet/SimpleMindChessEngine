# Simple Mind Chess Engine
# Bradley Thompson

# This is another helper that reads a pgn file, which is a representation for an entire game of chess that can be
# downloaded for any game played on chess.com. Basically, this script uses python-chess to parse the pgn file, builds
# out bitmap representations (using my ChessHelper) for each turn in the game, then adds it as labeled data (labeled as
# 1 for state associated with a winning side, 0 for state associated with losing side) to text file for dataset.

import sys
import chess.pgn
import numpy as np
from ChessHelper import ChessHelper as ch

DATASET_FILE_PATH = "./data/chess.data"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: missing pgn file.")

    pgn = open(sys.argv[1])
    chess_game = chess.pgn.read_game(pgn)

    result = chess_game.headers["Result"].split('-')
    winner = chess.WHITE if result[0] == '1' else chess.BLACK

    # Need to initialize np array with empty vector, needs chopping off after game processing
    board_state_history = np.zeros((768+1,))  # 786 input units, 1 data classification label

    board = chess.Board()
    for move in chess_game.mainline_moves():
        board.push(move)
        data_vector = np.reshape(ch.bitmap_representation(board), (768,))
        data_label = 1 if winner is chess.WHITE else -1
        data_line = np.append(data_vector, data_label)
        board_state_history = np.vstack((board_state_history, data_line))

    board_state_history = board_state_history[1:]  # Chop off that empty initialized vector

    # Getting rid of unwanted characters is annoyingly particular with numpy arrays...
    data = map(str, board_state_history)  # Make each row, corresponding to a given board state, a string.
    data = map(lambda x: x.replace('\n', ''), data)  #
    data = '\n'.join(data).replace('[', '').replace(']', '')

    dataset_file = open(DATASET_FILE_PATH, 'a')
    dataset_file.writelines(data)
    dataset_file.write('\n')  # Needed to make sure the script writes on a separate line on next run.
    dataset_file.close()
