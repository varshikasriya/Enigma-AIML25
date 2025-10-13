import math
import heapq


def sigmoid(x):
    return 1 / (1 + math.exp(-x))              

def relu(x):
    return max(0,x)                             

def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))             


def dotProduct(v1, v2):

    if len(v1) != len(v2):
        raise ValueError("nonono")
    
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def cosineSimilarity(v1, v2):
    dotp = dotProduct(v1, v2)

    x1 = math.sqrt(dotProduct(v1, v1))
    x2 = math.sqrt(dotProduct(v2, v2))

    if x1 == 0 or x2 == 0:
        return 0

    return dotp/(x1 * x2)

def l1Normalize(v):
    l1Norm = sum(abs(i) for i in v)

    if l1Norm == 0:
        return v
    
    return [i / l1Norm for i in v]

def l2Normalize(v):
    l2Norm = math.sqrt(sum(i**2 for i in v))


    if l2Norm == 0:
        return v
    

    return [i / l2Norm for i in v]

def minMaxNormalize(v):
    minVal = min(v)
    maxVal = max(v)
    rangeVal = maxVal - minVal
    
    if rangeVal == 0:
        return [0 for _ in v]
        
    return [(i - minVal) / rangeVal for i in v]


def heuristic(x, goal):
    return abs(goal - x)

def greedySearch(start, goal):
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start, [start]))
    visited = set()
    while pq:
        h, x, path = heapq.heappop(pq)
        if x == goal:
            return path
        if x in visited:
            continue
        visited.add(x)
        for nxt in [x + 1, x * 2]:
            if nxt not in visited and nxt <= goal * 2:
                heapq.heappush(pq, (heuristic(nxt, goal), nxt, path + [nxt]))
    return None


print("sigmoid(3):", sigmoid(3))
print("relu(21):", relu(21))
print("tanh(10):", tanh(10))


vecA = [1, 1, 2]
vecB = [1, 4, 7]

print("dot product: ", dotProduct(vecA, vecB))

print("cosine similarity:", cosineSimilarity(vecA, vecB))


vecC = [1, 2, 3]
vecD = [6, 7]
vecE = [1, 3, 9]
print("l1 normalize:", l1Normalize(vecC))
print("l2 normalize:", l2Normalize(vecD))
print("min max normalize:", minMaxNormalize(vecE))

start = 2
goal = 35
path = greedySearch(start, goal)
print(path)
