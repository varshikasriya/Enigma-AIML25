import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def dot_product(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length")
    return sum(a * b for a, b in zip(vec1, vec2))

def cosine_similarity(vec1, vec2):
    dot = dot_product(vec1, vec2)
    norm1 = math.sqrt(dot_product(vec1, vec1))
    norm2 = math.sqrt(dot_product(vec2, vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def l1_normalize(vec):
    norm = sum(abs(x) for x in vec)
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def l2_normalize(vec):
    norm = math.sqrt(sum(x**2 for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def min_max_normalize(vec):
    min_val = min(vec)
    max_val = max(vec)
    if max_val - min_val == 0:
        return [0 for _ in vec]
    return [(x - min_val) / (max_val - min_val) for x in vec]

def greedy_tic_tac_toe(board, player):
    def check_win(b, p):
        win_positions = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for pos in win_positions:
            if all(b[i] == p for i in pos):
                return True
        return False

    for i in range(9):
        if board[i] == 0:
            board[i] = player
            if check_win(board, player):
                board[i] = 0
                return i
            board[i] = 0

    opponent = 2 if player == 1 else 1
    for i in range(9):
        if board[i] == 0:
            board[i] = opponent
            if check_win(board, opponent):
                board[i] = 0
                return i
            board[i] = 0

    for i in range(9):
        if board[i] == 0:
            return i

    return -1
