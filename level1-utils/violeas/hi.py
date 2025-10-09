import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    dot = dot_product(v1, v2)
    mag1 = math.sqrt(dot_product(v1, v1))
    mag2 = math.sqrt(dot_product(v2, v2))
    return dot / (mag1 * mag2)

def normalize_l1(v):
    s = sum(abs(x) for x in v)
    return [x / s for x in v]

def normalize_l2(v):
    s = math.sqrt(sum(x ** 2 for x in v))
    return [x / s for x in v]

def normalize_minmax(v):
    mn, mx = min(v), max(v)
    return [(x - mn) / (mx - mn) for x in v]

