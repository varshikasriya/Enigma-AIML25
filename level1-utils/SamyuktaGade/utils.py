# level1-utils/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np

ArrayLike = Union[float, int, Sequence[float], Sequence[int], np.ndarray]

# ---------------------------
# Activation Functions
# ---------------------------

def sigmoid(x: ArrayLike) -> ArrayLike:
    """
    Numerically-stable sigmoid.
    For arrays: vectorized using logaddexp to prevent overflow.
    For scalars: returns float.
    """
    arr = np.asarray(x, dtype=np.float64)
    # Stable: σ(x) = exp(-logaddexp(0, -x))
    out = np.exp(-np.logaddexp(0.0, -arr))
    if out.ndim == 0:
        return float(out)
    return out


def relu(x: ArrayLike) -> ArrayLike:
    arr = np.asarray(x, dtype=np.float64)
    out = np.maximum(arr, 0.0)
    if out.ndim == 0:
        return float(out)
    return out


def tanh(x: ArrayLike) -> ArrayLike:
    arr = np.asarray(x, dtype=np.float64)
    out = np.tanh(arr)
    if out.ndim == 0:
        return float(out)
    return out


# ---------------------------
# Vector operations
# ---------------------------

def dot_product(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    1-D dot product with shape checks.
    """
    a = np.asarray(vec1, dtype=np.float64).ravel()
    b = np.asarray(vec2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))


def cosine_similarity(
    vec1: Sequence[float],
    vec2: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    """
    Cosine similarity with zero-vector safety.
    """
    a = np.asarray(vec1, dtype=np.float64).ravel()
    b = np.asarray(vec2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------
# Normalizations
# ---------------------------

def l1_normalize(x: ArrayLike, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    denom = np.sum(np.abs(arr), axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def l2_normalize(x: ArrayLike, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    denom = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def min_max_normalize(x: ArrayLike, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max to [0, 1], safe for constant slices.
    """
    arr = np.asarray(x, dtype=np.float64)
    x_min = np.min(arr, axis=axis, keepdims=True)
    x_max = np.max(arr, axis=axis, keepdims=True)
    denom = (x_max - x_min)
    denom = np.maximum(denom, eps)
    return (arr - x_min) / denom


# ---------------------------
# Tic-Tac-Toe Utilities
# ---------------------------
# Representation:
#  1 => current player
# -1 => opponent
#  0 => empty

Board = List[List[int]]

def greedy_tic_tac_toe_move(board: Board, player: int) -> Optional[Tuple[int, int]]:
    """
    Returns first empty spot (row, col).
    """
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return (i, j)
    return None


def _check_winner(board: Board) -> Optional[int]:
    """Return 1 if player 1 wins, -1 if player -1 wins, else None."""
    b = np.asarray(board, dtype=int)
    # Rows, cols
    for k in range(3):
        srow = b[k, :].sum()
        if srow == 3:
            return 1
        if srow == -3:
            return -1
        scol = b[:, k].sum()
        if scol == 3:
            return 1
        if scol == -3:
            return -1
    # Diagonals
    d1 = b.trace()
    d2 = np.fliplr(b).trace()
    if d1 == 3 or d2 == 3:
        return 1
    if d1 == -3 or d2 == -3:
        return -1
    return None


def _is_full(board: Board) -> bool:
    return all(cell != 0 for row in board for cell in row)


def _available_moves(board: Board) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]


@dataclass(frozen=True)
class MinimaxConfig:
    max_depth: int = 9
    alpha_beta: bool = True


def best_tic_tac_toe_move(
    board: Board,
    player: int,
    config: MinimaxConfig = MinimaxConfig(),
) -> Optional[Tuple[int, int]]:
    """
    Compute optimal move for 'player' using minimax with optional alpha-beta pruning.
    Returns (row, col) or None if no moves.
    """

    def minimax(state: Board, to_move: int, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int, int]]]:
        winner = _check_winner(state)
        if winner is not None:
            # Score from perspective of 'player'
            return (1.0 if winner == player else -1.0, None)
        if _is_full(state) or depth == 0:
            return (0.0, None)

        moves = _available_moves(state)
        if not moves:
            return (0.0, None)

        best_move: Optional[Tuple[int, int]] = None

        if to_move == player:
            # Maximizing
            best_score = -math.inf
            for (r, c) in moves:
                state[r][c] = to_move
                score, _ = minimax(state, -to_move, depth - 1, alpha, beta)
                state[r][c] = 0
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
                if config.alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
            return best_score, best_move
        else:
            # Minimizing
            best_score = math.inf
            for (r, c) in moves:
                state[r][c] = to_move
                score, _ = minimax(state, -to_move, depth - 1, alpha, beta)
                state[r][c] = 0
                if score < best_score:
                    best_score = score
                    best_move = (r, c)
                if config.alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
            return best_score, best_move

    _, move = minimax([row[:] for row in board], player, config.max_depth, -math.inf, math.inf)
    return move
