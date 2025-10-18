# level1-utils/test_utils.py
import math
import numpy as np
import pytest

from utils import (
    sigmoid, relu, tanh,
    dot_product, cosine_similarity,
    l1_normalize, l2_normalize, min_max_normalize,
    greedy_tic_tac_toe_move, best_tic_tac_toe_move, MinimaxConfig
)

def test_sigmoid_stability():
    assert math.isclose(sigmoid(1000.0), 1.0, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(sigmoid(-1000.0), 0.0, rel_tol=0, abs_tol=1e-15)
    x = np.array([-1000, -10, 0, 10, 1000], dtype=float)
    y = sigmoid(x)
    assert np.all(y >= 0) and np.all(y <= 1)

def test_relu_tanh_shapes():
    x = np.linspace(-3, 3, 7)
    assert relu(x).shape == x.shape
    assert tanh(x).shape == x.shape

def test_dot_and_cosine():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert dot_product(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0  # zero-vector safe

def test_normalizations():
    v = np.array([2.0, -1.0, 3.0])
    l1 = l1_normalize(v)
    assert np.isclose(np.sum(np.abs(l1)), 1.0)
    l2 = l2_normalize(v)
    assert np.isclose(np.linalg.norm(l2), 1.0)
    mm = min_max_normalize(np.array([2.0, 4.0, 6.0]))
    assert np.allclose(mm, [0.0, 0.5, 1.0])

    M = np.array([[1.0, 1.0], [1.0, 1.0]])
    mm_const = min_max_normalize(M, axis=1)
    assert np.all(mm_const == 0.0)

def test_tictactoe_greedy_and_minimax():
    # Greedy: first empty
    board = [
        [ 1,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ]
    g = greedy_tic_tac_toe_move(board, 1)
    assert g in [(0, 1), (1, 0), (1, 2), (2, 0), (2, 2)]

    # Minimax: winning move for player 1 at (0,2)
    board_win = [
        [ 1,  1,  0],
        [ 0, -1,  0],
        [ 0, -1,  0],
    ]
    move = best_tic_tac_toe_move(board_win, player=1, config=MinimaxConfig())
    assert move == (0, 2)

    # Minimax: player 1 must block opponent's imminent win
    board_block = [
        [ -1, -1,  0],
        [  1,  0,  0],
        [  0,  0,  1],
    ]
    move2 = best_tic_tac_toe_move(board_block, player=1, config=MinimaxConfig())
    assert move2 == (0, 2)
