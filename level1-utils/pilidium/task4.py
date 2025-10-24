def mismatches(source : list[list[int]], target : list[list[int]]) -> int:
    "Heuristic function that counts the number of mismatches between source and target 2D matrices."
    count = 0
    for srow, trow in zip(source, target):
        for sitem, titem in zip(srow, trow):
            if sitem != titem: count += 1
    return count

if __name__ == "__main__":

    source = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]

    target = [
        [4,2,7],
        [1,5,6],
        [3,9,8]
    ]

    print(mismatches(source, target))

    """
    Output:
    6
    """