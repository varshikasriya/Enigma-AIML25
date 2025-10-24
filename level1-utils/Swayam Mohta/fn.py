import math

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def relu(x: float) -> float:
    return max(0, x)

def tanh(x: float) -> float:
    return math.tanh(x)

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length.")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    dp = dot_product(v1, v2)
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dp / (mag1 * mag2)

def normalize_l1(vector):
    s = sum(abs(x) for x in vector)
    return [x / s for x in vector] if s != 0 else vector

def normalize_l2(vector):
    norm = math.sqrt(sum(x ** 2 for x in vector))
    return [x / norm for x in vector] if norm != 0 else vector

def normalize_minmax(vector):
    vmin, vmax = min(vector), max(vector)
    if vmax == vmin:
        return [0.0 for _ in vector]
    return [(x - vmin) / (vmax - vmin) for x in vector]

def greedy_move_tictactoe(board, player):
    win_patterns = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in win_patterns:
        cells = [board[a], board[b], board[c]]
        if cells.count(player) == 2 and cells.count(' ') == 1:
            return [a,b,c][cells.index(' ')]
    for i, cell in enumerate(board):
        if cell == ' ':
            return i
    return -1

if __name__ == "__main__":
    print("Activation Functions")
    print("sigmoid(0):", sigmoid(0))
    print("relu(5):", relu(5))
    print("tanh(1):", tanh(1))

    print("\nVector Operations")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("dot_product:", dot_product(v1, v2))
    print("cosine_similarity:", round(cosine_similarity(v1, v2), 4))

    print("\nNormalizations")
    v = [1, 2, 3, 4]
    print("L1:", normalize_l1(v))
    print("L2:", normalize_l2(v))
    print("Min-Max:", normalize_minmax(v))

    print("\nTic-Tac-Toe Heuristic")
    board = ['X', 'O', 'X',
             'O', 'X', ' ',
             ' ', ' ', 'O']
    player = 'X'
    move = greedy_move_tictactoe(board, player)
    print("Greedy move for player", player, ":", move)


#OUTPUT : 

# Activation Functions
# sigmoid(0): 0.5
# relu(5): 5
# tanh(1): 0.7615941559557649

# Vector Operations
# dot_product: 32
# cosine_similarity: 0.9746

# Normalizations
# L1: [0.1, 0.2, 0.3, 0.4]
# L2: [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214]
# Min-Max: [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]

# Tic-Tac-Toe Heuristic
# Greedy move for player X : 6
