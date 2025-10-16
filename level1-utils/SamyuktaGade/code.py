import numpy as np
from utils import (
    sigmoid, relu, tanh,
    dot_product, cosine_similarity,
    l1_normalize, l2_normalize, min_max_normalize,
    greedy_tic_tac_toe_move, best_tic_tac_toe_move, MinimaxConfig
)

# Activations
print(sigmoid(-10.0), sigmoid(10.0))
x = np.linspace(-8, 8, 1000)
y = sigmoid(x)     # vectorized

# Vector ops
a = [1, 2, 3]
b = [4, 5, 6]
print(dot_product(a, b))
print(cosine_similarity(a, b))

# Normalizations
v = np.array([2.0, -1.0, 3.0])
print(l1_normalize(v))
print(l2_normalize(v))
print(min_max_normalize(v))

# Tic-Tac-Toe
board = [
    [ 1,  1,  0],
    [ 0, -1,  0],
    [ 0, -1,  0],
]
print(best_tic_tac_toe_move(board, player=1, config=MinimaxConfig()))
