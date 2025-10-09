import math

# ----- Activation Functions -----

def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

def relu(x):
    """Compute the ReLU activation function (Rectified Linear Unit)."""
    return max(0, x)

def tanh(x):
    """Compute the hyperbolic tangent activation function."""
    return math.tanh(x)

# ----- Vector Operations -----

def dot_product(v1, v2):
    """Compute the dot product between two vectors."""
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    dot = dot_product(v1, v2)
    norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
    norm_v2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Avoid division by zero
    return dot / (norm_v1 * norm_v2)

# ----- Normalization Functions -----

def normalize_L1(vector):
    """Perform L1 normalization (sum of absolute values = 1)."""
    norm = sum(abs(x) for x in vector)
    return [x / norm for x in vector] if norm != 0 else vector

def normalize_L2(vector):
    """Perform L2 normalization (sum of squares = 1)."""
    norm = math.sqrt(sum(x ** 2 for x in vector))
    return [x / norm for x in vector] if norm != 0 else vector

def normalize_minmax(vector):
    """Perform Min-Max normalization (scale values to [0, 1])."""
    min_val, max_val = min(vector), max(vector)
    if max_val == min_val:
        return [0 for _ in vector]  # Avoid division by zero
    return [(x - min_val) / (max_val - min_val) for x in vector]

# ----- Heuristic Function (Example: Tic-Tac-Toe) -----

def greedy_tictactoe_heuristic(board, player):
    """
    Simple heuristic for Tic-Tac-Toe:
    - Checks for a winning move (2 in a row + 1 empty).
    - Otherwise, picks the first available empty cell.

    Parameters:
        board: list of 9 elements (3x3 flattened grid)
        player: 'X' or 'O'

    Returns:
        Index of the chosen move (0–8) or -1 if no move available.
    """
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]

    # Check for an immediate winning move
    for combo in win_combinations:
        values = [board[i] for i in combo]
        if values.count(player) == 2 and values.count(' ') == 1:
            return combo[values.index(' ')]

    # Otherwise, pick the first empty spot
    for i in range(len(board)):
        if board[i] == ' ':
            return i

    return -1  # No valid move found

# ----- Example Usage (Test Section) -----

if __name__ == "__main__":
    # Test activation functions
    print("Sigmoid(1):", sigmoid(1))
    print("ReLU(-5):", relu(-5))
    print("tanh(1):", tanh(1))

    # Test vector operations
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Dot product:", dot_product(v1, v2))
    print("Cosine similarity:", cosine_similarity(v1, v2))

    # Test normalization functions
    print("L1 normalization:", normalize_L1(v1))
    print("L2 normalization:", normalize_L2(v1))
    print("Min-max normalization:", normalize_minmax(v1))

    # Test heuristic function
    board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' ']
    move = greedy_tictactoe_heuristic(board, 'O')
    print("Greedy move for O:", move)
