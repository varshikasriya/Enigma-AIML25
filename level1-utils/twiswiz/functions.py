import math

#1.Activation Functions

def sigmoid(x):
    """Sigmoid activation function"""
    return 1/(1 + math.exp(-x))

def relu(x):
    """ReLU activation function"""
    return max(0, x)

def tanh(x):
    """Tanh activation function"""
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

#2.Vector Operations

def dot_product(v1, v2):
    """Compute dot product of two vectors"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length.")
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    dot = dot_product(v1, v2)
    mag1 = math.sqrt(dot_product(v1, v1))
    mag2 = math.sqrt(dot_product(v2, v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot/(mag1 * mag2)

#3.Normalization Methods

def l1_normalization(v):
    """L1 normalization (sum of absolute values = 1)"""
    s =sum(abs(x) for x in v)
    return [x / s for x in v] if s != 0 else v

def l2_normalization(v):
    """L2 normalization (Euclidean norm = 1)"""
    norm = math.sqrt(sum(x**2 for x in v))
    return [x / norm for x in v] if norm != 0 else v

def min_max_normalization(v):
    """Min-max normalization (scale between 0 and 1)"""
    v_min, v_max = min(v), max(v)
    if v_max == v_min:
        return [0.5 for _ in v]  # avoid divide by zero
    return [(x - v_min) / (v_max - v_min) for x in v]

#4.Heuristic Function (Tic-Tac-Toe)

def heuristic_tic_tac_toe(board, player):
    """
    Simple heuristic for Tic-Tac-Toe:
    - +10 if player can win next move
    - +5 for each row/column/diagonal with 2 player's marks and 1 empty
    - -5 for each opponent line with 2 marks and 1 empty
    - 0 otherwise
    """
    opponent = 'O' if player == 'X' else 'X'
    score = 0
    lines = [
        # Rows
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Columns
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diagonals
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]
    
    for line in lines:
        if line.count(player) == 2 and line.count(' ') == 1:
            score += 10
        elif line.count(player) == 1 and line.count(' ') == 2:
            score += 5
        elif line.count(opponent) == 2 and line.count(' ') == 1:
            score -= 5
    return score


if __name__ == "__main__":
    print("Sigmoid(1):", sigmoid(1))
    print("ReLU(-3):", relu(-3))
    print("Tanh(0.5):", tanh(0.5))
    
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Dot Product:", dot_product(v1, v2))
    print("Cosine Similarity:", cosine_similarity(v1, v2))
    
    v = [1, 2, 3]
    print("L1 Normalization:", l1_normalization(v))
    print("L2 Normalization:", l2_normalization(v))
    print("Min-Max Normalization:", min_max_normalization(v))
    
    board = [
        ['X', 'X', ' '],
        ['O', ' ', 'O'],
        [' ', ' ', 'X']
    ]
    print("Heuristic Score for X:", heuristic_tic_tac_toe(board, 'X'))