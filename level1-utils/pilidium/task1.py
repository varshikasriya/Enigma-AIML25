import math
from copy import deepcopy
from typing import Callable

def sigmoid(x : int | float) -> float:
    "Returns sigmoid of a number."
    return 1 / (1 + math.exp(-x))

def tanh(x : int | float) -> float:
    "Returns tanh of a number."
    plus, minus = math.exp(x), math.exp(-x)
    return (plus - minus) / (plus + minus)

def relu(x : int | float) -> float:
    "Returns relu of a number."
    return max(0, x)

# Eveyone's ignoring this possibility where the input is multidimensional.
# Does no one like element wise operators? :D
def operator(x : int | float | list, func : Callable[[int | float], float]) -> float | list | None:
    "Returns func of x where x is a number or a multidimensional list."
    if isinstance(x, (int, float)):
        return func(x)
    elif isinstance(x, list):
        x_copy = deepcopy(x)
        def operator_nested_list(obj : list) -> None:
            "Helper function to recursively apply func to innermost elements of x."
            for idx, item in enumerate(obj):
                if isinstance(item, (int, float)):
                    obj[idx] = func(item)
                elif isinstance(item, list):
                    operator_nested_list(item)
                else:
                    obj[idx] = None
        operator_nested_list(x_copy)
        return x_copy
    else:
        return None

if __name__ == "__main__":

    samples = [
        5,
        3.4,
        [1,2.5,3],
        [[14,15.6],[-0.04,100]],
        [1,[2,-3],[6,[-4,5]]]
    ]

    for sample in samples:
        print(f"Sigmoid of {sample} is {operator(sample, sigmoid)}")
        print(f"Tanh of {sample} is {operator(sample, tanh)}")
        print(f"ReLU of {sample} is {operator(sample, relu)}")

    """
    Output:
    Sigmoid of 5 is 0.9933071490757153
    Tanh of 5 is 0.999909204262595
    ReLU of 5 is 5
    Sigmoid of 3.4 is 0.9677045353015494
    Tanh of 3.4 is 0.9977749279342794
    ReLU of 3.4 is 3.4
    Sigmoid of [1, 2.5, 3] is [0.7310585786300049, 0.9241418199787566, 0.9525741268224334]
    Tanh of [1, 2.5, 3] is [0.7615941559557649, 0.9866142981514304, 0.9950547536867306]
    ReLU of [1, 2.5, 3] is [1, 2.5, 3]
    Sigmoid of [[14, 15.6], [-0.04, 100]] is [[0.9999991684719722, 0.9999998321172752], [0.4900013331200346, 1.0]]
    Tanh of [[14, 15.6], [-0.04, 100]] is [[0.9999999999986172, 0.9999999999999437], [-0.03997868031116358, 1.0]]
    ReLU of [[14, 15.6], [-0.04, 100]] is [[14, 15.6], [0, 100]]
    Sigmoid of [1, [2, -3], [6, [-4, 5]]] is [0.7310585786300049, [0.8807970779778823, 0.04742587317756678], [0.9975273768433653, [0.01798620996209156, 0.9933071490757153]]]
    Tanh of [1, [2, -3], [6, [-4, 5]]] is [0.7615941559557649, [0.964027580075817, -0.9950547536867306], [0.9999877116507956, [-0.9993292997390669, 0.999909204262595]]]
    ReLU of [1, [2, -3], [6, [-4, 5]]] is [1, [2, 0], [6, [0, 5]]]
    """