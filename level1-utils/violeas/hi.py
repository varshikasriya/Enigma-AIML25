import math

# 1️⃣ Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

# 2️⃣ Vector operations
def dot_product(v1, v2):
    return sum(a*b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    dot = dot_product(v1, v2)
    mag1 = math.sqrt(dot_product(v1, v1))
    mag2 = math.sqrt(dot_product(v2, v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

# 3️⃣ Normalization
def l1_normalize(v):
    total = sum(abs(x) for x in v)
    return [x/total for x in v] if total != 0 else v

def l2_normalize(v):
    total = math.sqrt(sum(x**2 for x in v))
    return [x/total for x in v] if total != 0 else v

def min_max_normalize(v):
    min_v, max_v = min(v), max(v)
    return [(x - min_v)/(max_v - min_v) for x in v] if max_v != min_v else v

# 4️⃣ Simple heuristic (greedy move for tic-tac-toe)
def greedy_move(board, player):
    opponent = 'O' if player == 'X' else 'X'

    # Try to win
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player
                if check_winner(board, player):
                    return (i, j)
                board[i][j] = ' '

    # Try to block opponent
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = opponent
                if check_winner(board, opponent):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '

    # Otherwise, pick first empty spot
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                return (i, j)

def check_winner(board, player):
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

# ✅ Example usage
if __name__ == "__main__":
    print("Sigmoid(0) =", sigmoid(0))
    print("ReLU(-3) =", relu(-3))
    print("Tanh(1) =", tanh(1))

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Dot product:", dot_product(v1, v2))
    print("Cosine similarity:", cosine_similarity(v1, v2))
    print("L1 normalize:", l1_normalize(v1))
    print("L2 normalize:", l2_normalize(v1))
    print("Min-max normalize:", min_max_normalize(v1))

    board = [['X', 'O', 'X'],
             [' ', 'O', ' '],
             [' ', ' ', ' ']]
    print("Greedy move for X:", greedy_move(board, 'X'))