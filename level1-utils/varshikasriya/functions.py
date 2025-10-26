import math


def sigmoid(x):  
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)


def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")

    dot_prod = dot_product(v1, v2)
    magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
    magnitude_v2 = math.sqrt(sum(x**2 for x in v2))

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    return dot_prod / (magnitude_v1 * magnitude_v2)


def l1_normalize(vector):
    l1_norm = sum(abs(x) for x in vector)
    if l1_norm == 0:
        return vector
    return [x / l1_norm for x in vector]

def l2_normalize(vector):
    l2_norm = math.sqrt(sum(x**2 for x in vector))
    if l2_norm == 0:
        return vector
    return [x / l2_norm for x in vector]

def minmax_normalize(vector):
    min_val = min(vector)
    max_val = max(vector)
    if max_val == min_val:
        return [0.5] * len(vector) 
    return [(x - min_val) / (max_val - min_val) for x in vector]



def tic_tac_toe_heuristic(board, player):
    opponent = 3 - player  


    def check_win(board, row, col, p):
        temp_board = [row[:] for row in board]
        temp_board[row][col] = p
        return is_winner(temp_board, p)


    def is_winner(board, p):

        for i in range(3):
            if all(board[i][j] == p for j in range(3)):
                return True
            if all(board[j][i] == p for j in range(3)):
                return True
        if all(board[i][i] == p for i in range(3)):
            return True
        if all(board[i][2-i] == p for i in range(3)):
            return True
        return False


    for i in range(3):
        for j in range(3):
            if board[i][j] == 0 and check_win(board, i, j, player):
                return (i, j)

    for i in range(3):
        for j in range(3):
            if board[i][j] == 0 and check_win(board, i, j, opponent):
                return (i, j)

    if board[1][1] == 0:
        return (1, 1)

    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for corner in corners:
        if board[corner[0]][corner[1]] == 0:
            return corner

    sides = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for side in sides:
        if board[side[0]][side[1]] == 0:
            return side

    return None 

if __name__ == "__main__":
    print("ACTIVATION FUNCTIONS")
    print(f"sigmoid(0) = {sigmoid(0):.4f}")
    print(f"sigmoid(2) = {sigmoid(2):.4f}")
    print(f"relu(-5) = {relu(-5)}")
    print(f"relu(3) = {relu(3)}")
    print(f"tanh(0) = {tanh(0):.4f}")
    print(f"tanh(1) = {tanh(1):.4f}")

    print("\nVECTOR OPERATIONS")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"Dot product = {dot_product(v1, v2)}")
    print(f"Cosine similarity = {cosine_similarity(v1, v2):.4f}")

    print("\nNORMALIZATION")
    vector = [1, 2, 3, 4]
    print(f"Original: {vector}")
    print(f"L1 normalized: {[round(x, 4) for x in l1_normalize(vector)]}")
    print(f"L2 normalized: {[round(x, 4) for x in l2_normalize(vector)]}")
    print(f"Min-Max normalized: {[round(x, 4) for x in minmax_normalize(vector)]}")

    print("\nTIC-TAC-TOE HEURISTIC")

    board = [
        [1, 0, 2],
        [0, 1, 0],
        [2, 0, 0]
    ]
    print("Board:")
    for row in board:
        print(row)
    move = tic_tac_toe_heuristic(board, 1)
    print(f"Best move for player 1: {move}")
