import math as m

# 1) Sigmoid = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + m.exp(-x))

# 2) ReLU = max(0, x)
def relu(x):
    return max(0, x)

# 3) Tanh = sinh(x) / cosh(x)
def tanh(x):
    return m.tanh(x)

# 4) Vector dot product
def dot(x, y):
    return sum(x[i] * y[i] for i in range(len(x)))

# 5) Cosine similarity
def cs(x, y):
    dotp = dot(x, y)
    magx = m.sqrt(dot(x, x))
    magy = m.sqrt(dot(y, y))
    return dotp / (magx * magy)

# 6A) Min-Max Normalization
def minmax(x):
    return [(i - min(x)) / (max(x) - min(x)) for i in x]

# 6B) L1 Normalization
def l1(x):
    s = sum(abs(i) for i in x)
    return [i / s for i in x]

# 6C) L2 Normalization
def l2(x):
    s = m.sqrt(sum(i**2 for i in x))
    return [i / s for i in x]

# 7) Heuristic: Tile mismatch in 3x3 grids
def heuristic(a, b):
    c = 0
    for i in range(3):
        for j in range(3):
            if a[i][j] != b[i][j]:
                c += 1
    return c
