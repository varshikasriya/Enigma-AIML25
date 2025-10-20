import math

# ====================================================================
# ----- 1) ACTIVATION FUNCTIONS (Used in Neural Networks) -----
# ====================================================================

def sigmoid(x: float) -> float:
    """
    Computes the Sigmoid activation function: 1 / (1 + e^(-x)).
    Scales the input value to a range between 0 and 1.
    """
    return 1.0 / (1.0 + math.exp(-x))

def relu(x: float) -> float:
    """
    Computes the Rectified Linear Unit (ReLU) activation function: max(0, x).
    Returns the input directly if it is positive, otherwise returns 0.
    """
    return max(0.0, x)

def tanh(x: float) -> float:
    """
    Computes the Hyperbolic Tangent (tanh) activation function.
    Scales the input value to a range between -1 and 1.
    """
    return math.tanh(x)

# ====================================================================
# ----- 2) VECTOR OPERATIONS (Used in NLP and Machine Learning) -----
# ====================================================================

def vector_dot_product(v1: list[float], v2: list[float]) -> float:
    """
    Computes the scalar product (dot product) of two equal-length vectors.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length for dot product.")
    
    # Use sum with zip for concise, efficient calculation
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Computes the cosine similarity between two vectors.
    Measures the cosine of the angle between them, indicating similarity.
    """
    dot_p = vector_dot_product(v1, v2)
    
    # Calculate vector magnitudes (L2 Norm)
    magnitude_v1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude_v2 = math.sqrt(sum(b ** 2 for b in v2))
    
    # Avoid division by zero if either vector is the zero vector
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    return dot_p / (magnitude_v1 * magnitude_v2)

# ====================================================================
# ----- 3) NORMALIZATION FUNCTIONS (Feature Scaling) -----
# ====================================================================

def normalize_minmax(vector: list[float]) -> list[float]:
    """
    Performs Min-Max Normalization (scaling) on a vector.
    Transforms values to a range of [0, 1].
    """
    min_val, max_val = min(vector), max(vector)
    
    # Handle case where all values are the same (denominator would be zero)
    if max_val == min_val:
        return [0.0] * len(vector)
        
    return [(x - min_val) / (max_val - min_val) for x in vector]

def normalize_L1(vector: list[float]) -> list[float]:
    """
    Performs L1 (Manhattan Norm) normalization.
    Scales the vector so the sum of the absolute values is 1.
    """
    l1_norm = sum(abs(x) for x in vector)
    
    if l1_norm == 0:
        return vector # Return original if zero vector
        
    return [x / l1_norm for x in vector]

def normalize_L2(vector: list[float]) -> list[float]:
    """
    Performs L2 (Euclidean Norm) normalization.
    Scales the vector so the square root of the sum of the squared values is 1.
    """
    l2_norm = math.sqrt(sum(x ** 2 for x in vector))
    
    if l2_norm == 0:
        return vector # Return original if zero vector

    return [x / l2_norm for x in vector]

# ====================================================================
# ----- 4) HEURISTIC FUNCTION (Example: 8-Puzzle Manhattan Distance) -----
# ====================================================================

def manhattan_distance_heuristic(current_grid: list[list[int]], goal_grid: list[list[int]]) -> int:
    """
    Calculates the Manhattan Distance heuristic for a 3x3 grid (like the 8-Puzzle).
    The distance is the sum of the moves required to move each tile from its
    current position to its position in the goal grid, ignoring the '0' tile.

    Note: Both grids must be 3x3 lists of lists.
    """
    distance = 0
    goal_positions = {}
    
    # 1. Map tile value to its (row, col) position in the goal grid
    for r in range(3):
        for c in range(3):
            goal_positions[goal_grid[r][c]] = (r, c)

    # 2. Calculate the total distance for the current grid
    for r in range(3):
        for c in range(3):
            tile_value = current_grid[r][c]
            
            # The '0' tile (blank space) is not counted
            if tile_value != 0:
                goal_r, goal_c = goal_positions[tile_value]
                
                # Manhattan Distance = |current_row - goal_row| + |current_col - goal_col|
                distance += abs(r - goal_r) + abs(c - goal_c)
                
    return distance

# ====================================================================
# ----- EXAMPLE USAGE (Optional but Recommended Test Section) -----
# ====================================================================

if __name__ == "__main__":
    print("--- Activation Functions ---")
    print(f"Sigmoid(0.5): {sigmoid(0.5):.4f}")
    print(f"ReLU(-10): {relu(-10.0):.1f}")
    print(f"Tanh(1.0): {tanh(1.0):.4f}")
    print("-" * 30)

    print("--- Vector Operations ---")
    vA = [1.0, 3.0, 5.0]
    vB = [2.0, 4.0, 6.0]
    print(f"Vector A: {vA}")
    print(f"Vector B: {vB}")
    print(f"Dot Product: {vector_dot_product(vA, vB)}")
    print(f"Cosine Similarity: {cosine_similarity(vA, vB):.4f}")
    print("-" * 30)

    print("--- Normalization ---")
    v_norm = [1.0, 2.0, 7.0]
    print(f"Original Vector: {v_norm}")
    print(f"Min-Max Norm: {normalize_minmax(v_norm)}")
    print(f"L1 Norm: {[f'{x:.4f}' for x in normalize_L1(v_norm)]}")
    print(f"L2 Norm: {[f'{x:.4f}' for x in normalize_L2(v_norm)]}")
    print("-" * 30)
    
    print("--- Heuristic Function (Manhattan Distance) ---")
    current = [[1, 2, 3], [4, 0, 5], [6, 7, 8]] # Tile 0 is at (1,1)
    goal =    [[1, 2, 3], [4, 5, 8], [6, 7, 0]] # Tile 0 is at (2,2)
    
    # Tile 5 is at (1, 2) in current, (1, 1) in goal -> distance 1
    # Tile 8 is at (2, 2) in current, (1, 2) in goal -> distance 1
    # Total distance should be 2
    
    dist = manhattan_distance_heuristic(current, goal)
    print(f"Current Grid: {current}")
    print(f"Goal Grid:    {goal}")
    print(f"Manhattan Distance: {dist}")
    print("-" * 30)