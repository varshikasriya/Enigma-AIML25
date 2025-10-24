import math

def norm_l1(x : list[int | float]) -> list[float]:
    "Returns L1 normalised x. This means the sum of absolute values of its elements will be 1."
    absum = sum(abs(i) for i in x)
    return [i/absum for i in x]

def norm_l2(x : list[int | float]) -> list[float]:
    "Returns L2 normalised x. This means the square root of the sum of squares of its elements will be 1."
    sqsum = math.sqrt(sum(i*i for i in x))
    return [i/sqsum for i in x]

def norm_min_max(x : list[int | float]) -> list[float]:
    "Returns (x-x_min)/(x_max-x_min)"
    x_max, x_min = max(x), min(x)
    diff = x_max - x_min
    return [(i - x_min)/diff for i in x]

if __name__ == "__main__":

    x = [-1.2,4,0,3.3]

    print(norm_l1(x))
    print(norm_l2(x))
    print(norm_min_max(x))

    """
    Output:
    [-0.1411764705882353, 0.47058823529411764, 0.0, 0.388235294117647]
    [-0.22545400718115802, 0.7515133572705267, 0.0, 0.6199985197481845]
    [0.0, 1.0, 0.23076923076923075, 0.8653846153846153]
    """