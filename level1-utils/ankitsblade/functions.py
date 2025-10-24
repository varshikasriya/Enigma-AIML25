import math

## Activation Functions

def sigmoid(x:float|list[float])->float|list[float]:
    """
    Compute the sigmoid for a given logit or list of logits
    Args:
        x (float | list[float]): The logit or list of logits to compute the sigmoid for
    Returns:
        float | list[float]: The sigmoid value(s)
    """
    if isinstance(x, list):
        return [1 / (1 + math.exp(-i)) for i in x]
    return 1 / (1 + math.exp(-x))

def tanh(x:float|list[float])->float|list[float]:
    """
    Compute the tanh for a given logit or list of logits 

    Args:
        x (float | list[float]): The logit or list of logits to compute the tanh for
    Returns:
        float | list[float]: The tanh value(s)
        
    """
    if isinstance(x,list):
        return [(math.exp(i) - math.exp(-i)) / (math.exp(i) + math.exp(-i)) for i in x]
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def relu(x:float|list[float])->float|list[float]:
    """
    Compute the ReLU for a given logit or list of logits

    Args:
        x (float | list[float]): The logit or list of logits to compute the ReLU for
    Returns:
        float | list[float]: The ReLU value(s)
    """
    if isinstance(x,list):
        return [max(0,i) for i in x]
    return max(0,x)

def softmax(x:list[float])->list[float]:
    """
    Compute the softmax for a given list of logits

    Args:
        x (list[float]): The list of logits to compute the softmax for
    Returns:
        list[float]: The softmax values
    """
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]

## Vector Operations

def dot_product(x:list[int],y:list[int]) -> int:
    """
    Compute the dot product of two vectors

    Args:
        x (list[int]): The first vector
        y (list[int]): The second vector
    Returns:
        int: The dot product of the two vectors
    """
    if len(x) != len(y):
        print("Vectors must be of same length")
    return sum(i*j for i,j in zip(x,y))

def mag(x:list[int])->float:
    return math.sqrt(sum(i * i for i in x))


def cosine_similarity(x:list[int],y:list[int])->float:
    """
    Compute the cosine similarity between two vectors

    Args:
        x (list[int]): The first vector
        y (list[int]): The second vector
    Returns:
        float: The cosine similarity between the two vectors
    """

    if len(x) != len(y):
        print("Vectors must be of same length")
    dot_prod = dot_product(x,y)
    mag_x = mag(x)
    mag_y = mag(y)
    if mag_x == 0 or mag_y == 0:
        return 0.0
    return dot_prod / (mag_x * mag_y)

## Normalisation functions

def L1_norm(x:list[int])->float:
    """
    Compute the L1 norm of a vector

    Args:
        x (list[int]): The vector to compute the L1 norm for
    Returns:
        float: The L1 norm of the vector
    """
    return sum(abs(i) for i in x)

def min_max_norm(x: list[float]) -> list[float]:
    """
    Min-max normalize a list of numbers to the range [0, 1].

    Args:
        x (list[float]): Input vector.
    Returns:
        list[float]: Normalized vector.
    """

    if not x:
        return []
    
    min_x = min(x)
    max_x = max(x)
    denom = max_x - min_x

    if denom ==0:
        return[0.0 for _ in x]
    return [(float(i)-min_x)/(denom) for i in x]


## Heuristic function

def hill_climb(x:float, step:float, max_iter:int)->float:
    """
    Perform hill climbing to find the maximum of a function

    Args:
        x (float): The starting point
        step (float): The step size
        max_iter (int): The maximum number of iterations
    Returns:
        float: The maximum value found
    """
    def f(x):
        return -(x-3)**2 + 5  # Example function with a maximum at x=3

    current_x = x
    current_y = f(current_x)

    for _ in range(max_iter):
        next_x = current_x + step
        next_y = f(next_x)

        if next_y > current_y:
            current_x = next_x
            current_y = next_y
        else:
            break

    return current_x

#Docstrings were written with the help of ChatGPT