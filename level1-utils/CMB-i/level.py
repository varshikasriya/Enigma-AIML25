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


# --- Simple Heuristic Function (Traveling Salesman: Nearest Neighbor) ---
def nearest_neighbor(distance_matrix, start=0):
    """
    Greedy TSP heuristic: always go to the nearest unvisited city.
    Args:
        distance_matrix: square matrix (list of lists) with non-negative distances.
                         distance_matrix[i][i] must be 0.
        start: starting city index (default 0)

    Returns:
        route: a list of city indices ending with the start city to form a cycle.
               Example: [0, 2, 1, 3, 0]
    """
    n = len(distance_matrix)
    if n == 0:
        return []

    # --- validation ---
    for row in distance_matrix:
        if len(row) != n:
            raise ValueError("Distance matrix must be square.")
    for i in range(n):
        if distance_matrix[i][i] != 0:
            raise ValueError("Diagonal (self-distance) must be 0.")
        for j in range(n):
            if distance_matrix[i][j] < 0:
                raise ValueError("Distances must be non-negative.")

    visited = [False] * n
    route = [start]
    visited[start] = True

    current = start
    for _ in range(n - 1):
        # choose nearest unvisited; deterministic tie-break on index
        candidates = [(j, distance_matrix[current][j]) for j in range(n) if not visited[j]]
        if not candidates:
            break
        next_city = min(candidates, key=lambda p: (p[1], p[0]))[0]
        route.append(next_city)
        visited[next_city] = True
        current = next_city

    # return to start to complete the tour
    route.append(start)
    return route


def route_length(distance_matrix, route):
    """Compute total length of a TSP route (assumes route is cyclic: ends at start)."""
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


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

if __name__ == "__main__":
    # ... keep your earlier sanity prints if you want ...

    # --- TSP demo ---
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    route = nearest_neighbor(dist, start=0)
    print("\nNearest-Neighbor TSP route:", route)
    print("Route length:", route_length(dist, route))
