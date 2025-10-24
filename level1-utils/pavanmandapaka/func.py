import math
#activation func

def sigmoid(x):
    """Sigmoid func"""
    return 1 / (1 + math.exp(-x))

def relu(x):
    """ReLU func"""
    return max(0, x)

def tanh(x):
    """Hyp Tang func"""
    return math.tanh(x)

#vector operation
def dot_product(v1, v2):
    """Dot product"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    """cosine similarity b/w two vectors."""
    dot = dot_product(v1, v2)
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

#normalization
def l1_normalize(vector):
    """L1 norm."""
    norm = sum(abs(x) for x in vector)
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def l2_normalize(vector):
    """ L2 norm."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def min_max_normalize(vector):
    """ Min-Max normalization."""
    min_val = min(vector)
    max_val = max(vector)
    if max_val == min_val:
        return [0 for _ in vector]
    return [(x - min_val) / (max_val - min_val) for x in vector]

#heuristic func
def simple_heuristic_move(board, player):
    """Simple greedy heuristic"""
    opponent = 'O' if player == 'X' else 'X'
    
    for i in range(9):
        if board[i] == ' ':
            temp = board.copy()
            temp[i] = player
            if check_winner(temp, player):
                return i
    
    for i in range(9):
        if board[i] == ' ':
            temp = board.copy()
            temp[i] = opponent
            if check_winner(temp, opponent):
                return i
    
    if board[4] == ' ':
        return 4
    
    for i in [0, 2, 6, 8]:
        if board[i] == ' ':
            return i
    
    for i in range(9):
        if board[i] == ' ':
            return i
    
    return -1  


def check_winner(board, player):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  
        [0, 4, 8], [2, 4, 6]              
    ]
    return any(all(board[pos] == player for pos in combo) for combo in wins)

if __name__ == "__main__":
    print("Sigmoid(0):", sigmoid(0))
    print("ReLU(-2):", relu(-2))
    print("tanh(1):", tanh(1))
    
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Dot Product:", dot_product(v1, v2))
    print("Cosine Similarity:", cosine_similarity(v1, v2))
    
    print("L1 Normalize:", l1_normalize(v1))
    print("L2 Normalize:", l2_normalize(v1))
    print("Min-Max Normalize:", min_max_normalize(v1))
    
    board = ['X', 'O', 'X',
             ' ', 'O', ' ',
             ' ', ' ', ' ']
    print("Best move for X:", simple_heuristic_move(board, 'X'))
