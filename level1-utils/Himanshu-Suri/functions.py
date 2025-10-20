import numpy as np
import heapq

# === FUNCTIONS ===

def sig(x):
    return 1 / (1 + np.exp(-x))

def reLu(x):
    return np.maximum(0, x)

def tanh(x):
    return 2 * sig(2 * x) - 1

def cosineSimilarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def norm(x):
    l1 = np.linalg.norm(x, 1)
    l2 = np.linalg.norm(x)
    min_max = (x - x.min()) / (x.max() - x.min())
    return l1, l2, min_max

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_verbose(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    step = 1
    while open_set:
        _, current = heapq.heappop(open_set)
        print(f"\n🔹 Step {step}: Exploring {current}")
        step += 1

        if current == goal:
            print("\n✅ Goal reached!\n")
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue  
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h = heuristic(neighbor, goal)
                    f = tentative_g + h
                    
                    print(f"  Neighbor {neighbor}: g={tentative_g}, h={h}, f={f}")
                    
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, neighbor))

    print("❌ No path found")
    return None


# === TEST CASES ===
if __name__ == "__main__":
    print("=== Testing Activation Functions ===")
    x = np.array([-2, -1, 0, 1, 2])
    print("Input:", x)
    print("Sigmoid:", sig(x))
    print("ReLU:", reLu(x))
    print("Tanh:", tanh(x))

    print("\n=== Testing Cosine Similarity ===")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("a =", a)
    print("b =", b)
    print("Cosine Similarity:", cosineSimilarity(a, b))

    print("\n=== Testing Norms ===")
    v = np.array([3, 4, 5])
    l1, l2, min_max = norm(v)
    print("Vector:", v)
    print("L1 Norm:", l1)
    print("L2 Norm:", l2)
    print("Min-Max Normalized:", min_max)

    print("\n=== Testing A* Algorithm ===")
    grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ]
    start = (0, 0)
    goal = (3, 3)
    path = astar_verbose(grid, start, goal)
    print("Path found:", path)
