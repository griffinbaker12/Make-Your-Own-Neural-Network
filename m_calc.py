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

X = [
    [0.9, 0.3, 0.4],
    [0.2, 0.8, 0.2],
    [0.1, 0.5, 0.6],
]
Y = [
    [0.9],
    [0.1],
    [0.8],
]
wih = np.array(X)
i = np.array(Y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X_ANS = np.dot(wih, i)
O = sigmoid(X_ANS)
print(O)


def mm(m1, m2):
    i, j, k = len(m1), len(m2), len(m2[0])
    if len(m1[0]) != k:
        print("Can't multiply this matrix")
        return
    result = [[0 for _ in range(k)] for _ in range(i)]
    for _i in range(i):
        for _j in range(j):
            for _k in range(k):
                result[_i][_k] += m1[_i][_j] * m2[_j][_k]
    return result


X = [[1, 2], [3, 4]]

Y = [[5, 6], [7, 8]]

print(mm(X, Y))
print(np.dot(X, Y))
