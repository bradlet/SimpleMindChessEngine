# SimpleMindChessEngine
A machine learning powered chess engine with the goal of staying as simple as possible.

#### Dev Info
> Developed by Bradley Thompson  
> Final project for CS 545 - Intro to Machine Learning (Fall 2020)  
> Prof: Anthony Rhodes

#### Machine learning in chess engines
A great lecture to watch:
https://www.youtube.com/watch?v=P0jd8AHwjXw&ab_channel=MachineLearningConference

Essentially, chess engines have been around for decades. At the turn of the millenium, 
they were basically just state space search algorithms, but more modern chess engines 
use machine learning to determine what board states are strong, then decide how to move
accordingly.

## Rough Notes for me as I work on project
Article on evaluating chess board state strength with machine learning:
https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

-	I’m leaning towards using a typed binary representation for the board.
-	I think that I want to make it so that I get just a random board history for a few games, break up the data into search trees of depth 2 or 3, where each node is the binary representation of the board at that state. This is going to be a very quickly growing tree and each node can have up to like 50 or 60 children.
-	For the sake of computation and simplicity I might make my tree search algorithm a super greedy algorithm. Just expand on the best child state from this node.
-	Will not consider resignations, will play until checkmate.
-	As I am not going to be generating a very large training set (If I am just going to basically generate it based on game play), I should consider some smart weight initialization that improves performance.
