"""
Level 1 Utils — adiavolo
Core math utils + a simple knapsack (greedy) heuristic.
"""

from typing import Sequence, List, Tuple
import math

# -------- Activations (scalar) --------


def sigmoid(x: float) -> float:
    """Numerically-stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def relu(x: float) -> float:
    """ReLU."""
    return x if x > 0.0 else 0.0


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)

# -------- Vector Ops --------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product; sizes must match."""
    if len(a) != len(b):
        raise ValueError(f"dot: length mismatch {len(a)} != {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise ValueError("dot: non-finite value")
        s += xf * yf
    return s


def _l2(x: Sequence[float]) -> float:
    """L2 norm."""
    s = 0.0
    for v in x:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("L2: non-finite value")
        s += fv * fv
    return math.sqrt(s)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; undefined for zero-norm."""
    if len(a) != len(b):
        raise ValueError(f"cosine: length mismatch {len(a)} != {len(b)}")
    na, nb = _l2(a), _l2(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine: zero-norm vector")
    return dot(a, b) / (na * nb)

# -------- Normalization --------


def normalize_l1(x: Sequence[float]) -> List[float]:
    """L1: sum(|x|)=1."""
    s = sum(abs(float(v)) for v in x)
    if s == 0.0:
        raise ValueError("L1: sum(|x|)=0")
    return [float(v) / s for v in x]


def normalize_l2(x: Sequence[float]) -> List[float]:
    """L2: ||x||=1."""
    n = _l2(x)
    if n == 0.0:
        raise ValueError("L2: ||x||=0")
    return [float(v) / n for v in x]


def normalize_minmax(
    x: Sequence[float],
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> List[float]:
    """Min–max to [low, high]."""
    low, high = float(feature_range[0]), float(feature_range[1])
    if not (low < high):
        raise ValueError("minmax: low < high required")
    xs = [float(v) for v in x]
    mn, mx = min(xs), max(xs)
    if mx == mn:
        raise ValueError("minmax: all values equal")
    scale = (high - low) / (mx - mn)
    return [low + (v - mn) * scale for v in xs]

# -------- Heuristic: Knapsack (Greedy by value/weight) --------


Item = Tuple[str, float, float]  # (name, value, weight)


def heuristic_knapsack_greedy(items: Sequence[Item], capacity: float):
    """
    Greedy 0/1 knapsack by value/weight.
    Returns: (selected_items, total_value, total_weight)
    """
    if capacity < 0:
        raise ValueError("capacity must be >= 0")

    norm: List[Tuple[int, str, float, float]] = []
    for i, it in enumerate(items):
        if len(it) != 3:
            raise ValueError("item must be (name, value, weight)")
        name, value, weight = it
        if not isinstance(name, str) or not name.strip():
            raise ValueError("item name empty")
        v, w = float(value), float(weight)
        if v < 0:
            raise ValueError(f"value < 0 for '{name}'")
        if w <= 0:
            raise ValueError(f"weight <= 0 for '{name}'")
        norm.append((i, name.strip(), v, w))

    # Sort: ratio desc, value desc, weight asc, original index asc
    def ratio(v: float, w: float) -> float: return v / w
    norm.sort(key=lambda t: (ratio(t[2], t[3]),
              t[2], -t[3], -t[0]), reverse=True)

    selected: List[Item] = []
    rem = float(capacity)
    tv = 0.0
    tw = 0.0
    for _, name, v, w in norm:
        if w <= rem:
            selected.append((name, v, w))
            rem -= w
            tv += v
            tw += w
    return selected, tv, tw


__all__ = [
    "sigmoid", "relu", "tanh",
    "dot", "cosine_similarity",
    "normalize_l1", "normalize_l2", "normalize_minmax",
    "heuristic_knapsack_greedy",
]
