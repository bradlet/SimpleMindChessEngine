# SimpleMindChessEngine
A machine learning powered chess engine with the goal of staying as simple as possible.

###### Note:
For simplicities sake, I'm not going to start this as a full chess-playing engine. It's going 
to simply be a multi-layered neural network implementation that evaluates how strong a board 
position is.

Later on, this can be expanded to then evaluate which possible move should be chosen, and then
it can be transformed into a full-blown chess engine.

#### Dev Info
> Developed by Bradley Thompson  
> Final project for CS 545 - Intro to Machine Learning (Fall 2020)  
> Prof: Anthony Rhodes

### Machine learning in chess engines
A great lecture to watch:
https://www.youtube.com/watch?v=P0jd8AHwjXw&ab_channel=MachineLearningConference

Essentially, chess engines have been around for decades. At the turn of the millenium, 
they were basically just state space search algorithms, but more modern chess engines 
use machine learning to determine what board states are strong, then decide how to move
accordingly.

## Data Representation
Dataset constructed from chess.com data using DatasetBuilder.py. The data is a flattened bitmap
representation of a chessboard: 1 board state per row, with the final column holding the label.
Currently using label of winning state ('1'), and losing state ('-1') to keep it simple.

I didn't come up with the bitmap representation concept, pulled from this article on evaluating 
chess board state strength with machine learning:
https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

## python-chess
I'm using python-chess so that I don't have to worry about the logic behind a lot of
Chess rules. The library is great, [check it out!](https://python-chess.readthedocs.io/en/latest/)

    # Example of three ways to make moves
    move = chess.Move(from_square=chess.E2, to_square=chess.E4)
    board.push(move)
    move2 = chess.Move.from_uci("e7e5")
    board.push(move2)
    # The board will accept non-legal moves. Have to be careful about that.
    # Board also has no concept of which turn it is as far as push_san() is concerned.
    board.push_san("exf4") 
