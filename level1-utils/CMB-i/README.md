# Utils
Implemented Functions

1. Activation Functions

These functions are fundamental building blocks in neural networks.

| Function       | Description                                            | Formula                         |
| -------------- | ------------------------------------------------------ | ------------------------------- |
| **sigmoid(x)** | Maps input to (0, 1). Useful for probabilities.        | 1 / (1 + e^(-x))                |
| **relu(x)**    | Rectified Linear Unit. Common activation in deep nets. | max(0, x)                       |
| **tanh(x)**    | Scaled sigmoid between (-1, 1). Zero-centered.         | (e^x – e^(-x)) / (e^x + e^(-x)) |


Usage:

sigmoid(0)  # → 0.5

relu(-3)    # → 0

tanh(1)     # → 0.7615


2. Vector Operations

Function	Description

| Function                      | Description                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| **vector_dot(v1, v2)**        | Computes dot product between two equal-length vectors.                                     |
| **cosine_similarity(v1, v2)** | Measures orientation similarity between vectors (1 = identical direction, 0 = orthogonal). |


Example:

v1, v2 = [1, 2, 3], [4, 5, 6]

vector_dot(v1, v2)         # 32

cosine_similarity(v1, v2)  # 0.9746


These operations are the foundation of similarity search, word embeddings, and recommendation systems.

3. Normalization Methods

Normalization ensures data is on a comparable scale before feeding into models.

| Method                    | Description                        | Example Output                        |
| ------------------------- | ---------------------------------- | ------------------------------------- |
| **L1 normalization**      | Sum of absolute values = 1.        | `[0.066, 0.133, 0.200, 0.266, 0.333]` |
| **L2 normalization**      | Square root of sum of squares = 1. | `[0.134, 0.268, 0.402, 0.537, 0.671]` |
| **Min-Max normalization** | Scales data to [0, 1] range.       | `[0, 0.25, 0.5, 0.75, 1]`             |


4. Heuristic Algorithm: Traveling Salesman (Nearest Neighbor)

A heuristic is a rule-of-thumb algorithm — fast, approximate, and practical when exact solutions are expensive.

nearest_neighbor(distance_matrix, start=0)

Greedy heuristic that:
- Starts at a given city.
- Repeatedly visits the nearest unvisited city.
- Returns to the start to complete the cycle.

route_length(distance_matrix, route)

Calculates the total cost (distance) of a route.

Example:

dist = [
    [0,10,15,20],
    
  [10,0,35,25],
    
  [15,35,0,30],
    
   [20,25,30,0]
    
]

route = nearest_neighbor(dist, start=0)

print("Route:", route)

print("Total distance:", route_length(dist, route))


Output:

Route: [0, 1, 3, 2, 0]

Total distance: 80
