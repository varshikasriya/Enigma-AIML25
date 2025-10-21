import math

# Sigmoid curve = 1/(1 + e^-x)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ReLU (Rectified Linear Unit) = max(0, x). It outputs x if x is positive; otherwise, it outputs 0.
def relu(x):
    return max(0, x)

# Tanh (Hyperbolic Tangent) = (e^x - e^-x) / (e^x + e^-x)
def tanh(x):    
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

# Dot Product of two vectors
def dot_product(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length")
    result = 0
    for i in range(len(vec_a)):
        result += vec_a[i] * vec_b[i]
    return result

# Cosine Similarity between two vectors
def cosine_similarity(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length")
    
    dot_prod = dot_product(vec_a, vec_b)
    magnitude_a = math.sqrt(dot_product(vec_a, vec_a))
    magnitude_b = math.sqrt(dot_product(vec_b, vec_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("One or both vectors have zero magnitude")
    
    return dot_prod / (magnitude_a * magnitude_b)

