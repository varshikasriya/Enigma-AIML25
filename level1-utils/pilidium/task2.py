import math

def vector_dot_product(vec1 : list[int | float], vec2 : list[int | float]) -> float | None:
    "Returns dot product of same shape vectors."
    if len(vec1) != len(vec2): return None
    result = 0
    for i1, i2 in zip(vec1, vec2):
        result += i1 * i2
    return result

def cosine_similarity(vec1 : list[int | float], vec2 : list[int | float]) -> float | None:
    "Returns cosine similarity of same shape vectors."
    if len(vec1) != len(vec2): return None
    return vector_dot_product(vec1, vec2) / (math.sqrt(vector_dot_product(vec1, vec1)) * math.sqrt(vector_dot_product(vec2, vec2)))

if __name__ == "__main__":

    v1 = [1, 0, -2.3]
    v2 = [-3.12, 10, 2]

    print(vector_dot_product(v1, v2))
    print(cosine_similarity(v1, v2))

    """
    Output:
    -7.72
    -0.28863304905634257
    """