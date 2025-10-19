import math
# Activation Functions

def sigmoid(x):
    """Sigmoid"""
    return 1 / (1 + math.exp(-x))

def relu(x):
    """ReLU"""
    return max(0, x)

def tanh(x):
    """Tanh"""
    return math.tanh(x)

# Vector Operations

def dot_product(v1, v2):
    """Dot Product"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    """Cosine Similarity"""
    dot = dot_product(v1, v2)
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

# Normalization Functions

def normalize_l1(v):
    """L1 normalization."""
    norm = sum(abs(x) for x in v)
    return [x / norm for x in v] if norm != 0 else v

def normalize_l2(v):
    """L2 normalization."""
    norm = math.sqrt(sum(x ** 2 for x in v))
    return [x / norm for x in v] if norm != 0 else v

def normalize_minmax(v):
    """Min-max normalization."""
    min_v, max_v = min(v), max(v)
    if max_v == min_v:
        return [0 for _ in v]
    return [(x - min_v) / (max_v - min_v) for x in v]

# Heuristic Function

def tic_tac_toe_heuristic(board, player):
    """
      Heuristic for Tic-Tac-Toe game state.
    """
    opponent = 'O' if player == 'X' else 'X'
    lines = [
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]],
    ]

    score = 0
    for line in lines:
        if opponent not in line:
            score += 1
        if player not in line:
            score -= 1
    return score



if __name__ == "__main__":
    print("=== Activation Functions ===")
    print(f"sigmoid(0): {sigmoid(0)}")
    print(f"relu(5): {relu(5)}")
    print(f"tanh(1): {tanh(1)}")
    
    print("\n=== Vector Operations ===")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"dot_product(v1, v2): {dot_product(v1, v2)}")
    print(f"cosine_similarity(v1, v2): {cosine_similarity(v1, v2):.4f}")
    
    print("\n=== Normalization Functions ===")
    v = [1, 2, 3, 4]
    print(f"Original vector: {v}")
    print(f"L1 normalization: {normalize_l1(v)}")
    print(f"L2 normalization: {normalize_l2(v)}")
    print(f"Min-Max normalization: {normalize_minmax(v)}")
    
    print("\n=== Tic-Tac-Toe Heuristic ===")
    board = ['X', 'O', 'X', 
             'O', 'X', ' ', 
             ' ', ' ', 'O']
    print("Board state:")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---|---|---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---|---|---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print(f"Heuristic score for 'X': {tic_tac_toe_heuristic(board, 'X')}")
    print(f"Heuristic score for 'O': {tic_tac_toe_heuristic(board, 'O')}")