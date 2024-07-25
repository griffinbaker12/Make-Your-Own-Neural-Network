import numpy as np


def mm(A, B):
    if len(A[0]) != len(B):
        print("Invalid matrix dimensions.")
        return

    output_rows, output_cols = len(A), len(B[0])
    output = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    i, j, k = len(A), len(B), len(B[0])

    for _i in range(i):
        for _j in range(j):
            for _k in range(k):
                output[_i][_k] += A[_i][_j] * B[_j][_k]

    return output


A = [
    [1, 2],
    [3, 4],
]
B = [
    [5, 6],
    [7, 8],
]

print(mm(A, B))
print()
print(np.dot(A, B))
