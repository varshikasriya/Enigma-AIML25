import math

# --- Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

def relu(x):
    """ReLU activation function."""
    return max(0, x)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return math.tanh(x)


# --- Vector Operations ---
def vector_dot(v1, v2):
    """Compute dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of same length.")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = vector_dot(v1, v2)
    norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
    norm_v2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot / (norm_v1 * norm_v2)


# --- Normalization Functions ---
def normalize(data, method="l2"):
    """Normalize a list of numbers using L1, L2, or Min-Max normalization."""
    if not data:
        return []

    if method.lower() == "l1":
        norm = sum(abs(x) for x in data)
        return [x / norm for x in data] if norm != 0 else data

    elif method.lower() == "l2":
        norm = math.sqrt(sum(x ** 2 for x in data))
        return [x / norm for x in data] if norm != 0 else data

    elif method.lower() in ["min-max", "minmax"]:
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [0 for _ in data]
        return [(x - min_val) / (max_val - min_val) for x in data]

    else:
        raise ValueError("Unknown normalization method. Use 'l1', 'l2', or 'min-max'.")


# --- Simple Heuristic Function (Tic-Tac-Toe) ---
def greedy_tic_tac_toe(board, player):
    """
    Greedy move for Tic-Tac-Toe.
    board: list of 9 elements ('X', 'O', or ' ')
    player: 'X' or 'O'
    Returns: index (0-8) of the best move.
    """
    for i in range(9):
        if board[i] == ' ':
            temp = board.copy()
            temp[i] = player
            if is_winner(temp, player):
                return i

    opponent = 'O' if player == 'X' else 'X'
    for i in range(9):
        if board[i] == ' ':
            temp = board.copy()
            temp[i] = opponent
            if is_winner(temp, opponent):
                return i

    for i in range(9):
        if board[i] == ' ':
            return i

    return -1

def is_winner(board, player):
    """Check if player has won."""
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    return any(all(board[i] == player for i in combo) for combo in wins)


# --- Test Run ---
if __name__ == "__main__":
    print("Sigmoid(0):", sigmoid(0))
    print("ReLU(-5):", relu(-5))
    print("tanh(1):", tanh(1))

    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("\nDot Product:", vector_dot(v1, v2))
    print("Cosine Similarity:", cosine_similarity(v1, v2))

    data = [1, 2, 3, 4, 5]
    print("\nL1 Normalization:", normalize(data, "l1"))
    print("L2 Normalization:", normalize(data, "l2"))
    print("Min-Max Normalization:", normalize(data, "min-max"))

    board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' ']
    move = greedy_tic_tac_toe(board, 'X')
    print("\nGreedy Tic-Tac-Toe move for X:", move)
