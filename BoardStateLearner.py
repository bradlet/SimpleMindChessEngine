# Simple Mind Chess Engine
# Bradley Thompson

# Multi-Layered Neural Network implementation
class BoardStateLearner:

    # This is not trying to be a generalizable implementation.
    # Operates specifically on a dataset where each row is a bitmap representation vector for a chess board of len. 768.
    def __init__(self, dataset):
        self.data = dataset
