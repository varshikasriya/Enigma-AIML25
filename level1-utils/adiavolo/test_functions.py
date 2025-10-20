import math
from functions import (
    sigmoid, relu, tanh,
    dot, cosine_similarity,
    normalize_l1, normalize_l2, normalize_minmax,
    heuristic_knapsack_greedy,
)

# --- tiny print helpers ---


def hr(title=""):
    print("\n" + "=" * 12 + (" " + title if title else "") + "\n")


def show(name, value):
    print(f"{name}: {value}")


def approx(a, b, eps=1e-9):
    return abs(a - b) <= eps

# ----------------- tests with prints -----------------


def test_activations():
    hr("Activations")
    xvals = [-3.0, 0.0, 2.0]
    sig = [sigmoid(x) for x in xvals]
    rel = [relu(x) for x in xvals]
    th = [tanh(x) for x in xvals]
    show("x", xvals)
    show("sigmoid(x)", [round(v, 6) for v in sig])
    show("relu(x)", rel)
    show("tanh(x)", [round(v, 6) for v in th])

    assert approx(sigmoid(0.0), 0.5)
    assert sigmoid(10) > 0.9999 and sigmoid(-10) < 0.0001
    assert relu(-3.0) == 0.0 and relu(4.2) == 4.2
    assert approx(tanh(0.0), 0.0)


def test_vectors():
    hr("Vector Ops")
    a, b = [1, 2, 3], [4, 5, 6]
    d = dot(a, b)
    c_same = cosine_similarity([1, 1], [2, 2])
    c_orth = cosine_similarity([1, 0], [0, 1])
    show("a", a)
    show("b", b)
    show("dot(a,b)", d)
    show("cosine([1,1],[2,2])", round(c_same, 6))
    show("cosine([1,0],[0,1])", round(c_orth, 6))

    assert approx(d, 32.0)
    assert approx(c_same, 1.0)
    assert approx(c_orth, 0.0)
    try:
        cosine_similarity([0, 0], [1, 2])
        assert False, "should raise on zero-norm"
    except ValueError:
        pass


def test_normalization():
    hr("Normalization")
    x_l1 = [1, -1, 2]
    x_l2 = [3, 4]
    x_mm = [0, 5, 10]
    l1 = normalize_l1(x_l1)
    l2 = normalize_l2(x_l2)
    mm = normalize_minmax(x_mm)
    show("L1 input", x_l1)
    show("L1 output", [round(v, 6) for v in l1])
    show("L2 input", x_l2)
    show("L2 output", [round(v, 6) for v in l2])
    show("MinMax input", x_mm)
    show("MinMax output", [round(v, 6) for v in mm])

    assert l1 == [0.25, -0.25, 0.5]
    assert all(approx(a, b) for a, b in zip(l2, [0.6, 0.8]))
    assert all(approx(a, b) for a, b in zip(mm, [0.0, 0.5, 1.0]))

    # error cases
    for bad in ([0, 0, 0],):
        try:
            normalize_l1(bad)
            assert False
        except ValueError:
            pass
    for bad in ([0, 0],):
        try:
            normalize_l2(bad)
            assert False
        except ValueError:
            pass
    for bad in ([7, 7, 7],):
        try:
            normalize_minmax(bad)
            assert False
        except ValueError:
            pass
    try:
        normalize_minmax([1, 2], feature_range=(1.0, 1.0))
        assert False
    except ValueError:
        pass


def test_knapsack_greedy():
    hr("Heuristic: Knapsack (Greedy)")
    items = [
        ("Tent", 10, 5),
        ("Food", 8, 4),
        ("Water", 7, 3),
        ("First Aid", 6, 2),
        ("Camera", 5, 4),
        ("Power Bank", 4, 2),
    ]
    cap = 15
    selected, total_value, total_weight = heuristic_knapsack_greedy(items, cap)

    # pretty print
    print("Items (name, value, weight):")
    for it in items:
        print("  ", it)
    show("Capacity", cap)
    print("Selected (order):")
    for it in selected:
        print("  ", it)
    show("Totals", {"value": total_value, "weight": total_weight})

    # asserts
    # order depends on tie-breaks; this matches the provided implementation.
    assert [n for (n, _, _) in selected] == [
        "First Aid", "Water", "Tent", "Food"]
    assert total_weight <= cap
    assert total_value == 31.0
    assert total_weight == 14.0

# ----------------- runner -----------------


if __name__ == "__main__":
    test_activations()
    test_vectors()
    test_normalization()
    test_knapsack_greedy()
    hr("ALL TESTS PASSED")
