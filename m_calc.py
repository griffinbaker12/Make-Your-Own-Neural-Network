import numpy as np

m1 = [
    [0.9, 0.3],
    [0.2, 0.8],
]
m2 = [
    [0.9],
    [0.1],
]


def mat_mul(m1, m2):
    result = [0, 0]
    for i in range(2):
        for k in range(2):
            result[i] += m1[i][k] * m2[k][0]
    return result


print(mat_mul(m1, m2))


def matrix_multiply(A, B):
    result = [0, 0, 0]
    for i in range(3):  # For each row of the result
        for k in range(3):  # For each element in the row of A / column of B
            result[i] += A[i][k] * B[k][0]
    return result


A3 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

B3 = [
    [10],
    [11],
    [12],
]
print(matrix_multiply(A3, B3))
print(np.dot(A3, B3))

# Example usage

A = np.array(
    [
        [0.9, 0.3],
        [0.2, 0.8],
    ]
)

B = np.array(
    [
        [0.9],
        [0.1],
    ]
)

print(np.dot(A, B))
