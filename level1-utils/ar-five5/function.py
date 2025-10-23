import math

## Sigmoid, Relu & Tanh functions
# Sigmoid func
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Relu func
def relu(x):
    return max(0, x)

# Tanh func
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

## Dot Product & Cosine Similarity
# Dot Product func
def dotproduct(a, b):
    dotp = 0
    for i in range(len(a)):
        dotp = dotp + a[i] * b[i]
    return dotp

# Cosine Similarity
def coSim(a, b):
    dotp = dotproduct(a, b)
    x1 = math.sqrt(dotproduct(a, a))
    x2 = math.sqrt(dotproduct(b, b))
    if x1 == 0 or x2 == 0:
        return 0
    return dotp / (x1 * x2)

## L1 , L2 & Min-Max normalization
# L1 normalization
def l1_normalize(x):
    total = sum(abs(i) for i in x)
    for i in range(len(x)):
        x[i] = x[i] / total
        x[i] = round(x[i], 4)
    return x

# L2 normalization
def l2_normalize(x):
    total = math.sqrt(sum(i * i for i in x))
    for i in range(len(x)):
        x[i] = x[i] / total
        x[i] = round(x[i], 4)
    return x

# Min-Max normalization
def min_max_normalize(x):
    min_val = min(x)
    max_val = max(x)
    for i in range(len(x)):
        x[i] = (x[i] - min_val) / (max_val - min_val)
        x[i] = round(x[i], 4)
    return x

## Heuristic Functions
# Manhattan distance
def manhattan_distance(a, b):
    total = sum(abs(a[i] - b[i]) for i in range(len(a)))
    return round(total, 4)

# Euclidean distance
def euclidean_distance(a, b):
    total = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
    return round(total, 4)
