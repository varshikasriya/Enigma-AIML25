import math
from typing import List

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def relu(x: float) -> float:
    return max(0.0, x)

def tanh(x: float) -> float:
    return math.tanh(x)

def dot_product(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    return sum(a * b for a, b in zip(vec1, vec2))

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_prod = dot_product(vec1, vec2)
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_prod / (norm1 * norm2)

def l1_normalize(vector: List[float]) -> List[float]:
    l1_norm = sum(abs(x) for x in vector)
    if l1_norm == 0:
        return vector
    return [x / l1_norm for x in vector]

def l2_normalize(vector: List[float]) -> List[float]:
    l2_norm = math.sqrt(sum(x * x for x in vector))
    if l2_norm == 0:
        return vector
    return [x / l2_norm for x in vector]

def min_max_normalize(vector: List[float]) -> List[float]:
    if not vector:
        return vector
    min_val = min(vector)
    max_val = max(vector)
    if max_val == min_val:
        return [0.5] * len(vector)
    return [(x - min_val) / (max_val - min_val) for x in vector]

def greedy_tic_tac_toe_move(board: List[List[str]], player: str) -> tuple:
    opponent = 'O' if player == 'X' else 'X'
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player
                if check_win(board, player):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = opponent
                if check_win(board, opponent):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '
    
    if board[1][1] == ' ':
        return (1, 1)
    
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for i, j in corners:
        if board[i][j] == ' ':
            return (i, j)
    
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for i, j in edges:
        if board[i][j] == ' ':
            return (i, j)
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                return (i, j)
    
    return (-1, -1)

def check_win(board: List[List[str]], player: str) -> bool:
    for i in range(3):
        if all(board[i][j] == player for j in range(3)):
            return True
        if all(board[j][i] == player for j in range(3)):
            return True
    
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2-i] == player for i in range(3)):
        return True
    
    return False