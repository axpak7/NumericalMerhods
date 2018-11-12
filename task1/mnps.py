"""
    Метод наискорейшего покоординатного спуска
"""

import numpy as np


def polynomValue(X):
    """
    Нахождение значения полинома
    :param X: задание неизвестных переменных x, y, z
    :return: значение полинома
    """
    x = X[0]
    y = X[1]
    z = X[2]
    return 2 * x ** 2 + 3.1 * y ** 2 + 4.1 * z ** 2 + x * y - y * z + x * z + x - 2 * y + 3 * z + 1


def stepValue(A, x, b, e):
    """
    Вычисление шага m
    :param A: матрица А
    :param x: вектор х
    :param b: вектор b
    :param e: вектор направления спуска
    :return:
    """
    e_transpose = e.transpose()
    numerator = np.dot(e_transpose, (np.dot(A, x) + b))
    denominator = np.dot(e_transpose, np.dot(A, e))
    return -numerator / denominator


# constants
A = np.array([[4, 1, 1], [1, 6.2, -1], [1, -1, 8.2]])
b = np.array([[1], [-2], [3]])
epsilon = 0.0000001
# first member
x = np.array([[0], [0], [0]])
new_f = polynomValue(x)
f = 0
# Direction of descent
e1 = np.array([[1], [0], [0]])
e2 = np.array([[0], [1], [0]])
e3 = np.array([[0], [0], [1]])
print("iteration: 1")
print("vector x = \n", x)
print("polynom value = ", new_f, "\n")
i = 2
while abs(f - new_f) > epsilon:
    print("iteration: ", i)
    f = new_f
    m1 = stepValue(A, x, b, e1)
    m2 = stepValue(A, x, b, e2)
    m3 = stepValue(A, x, b, e3)
    x1 = x + m1 * e1
    x2 = x + m2 * e2
    x3 = x + m3 * e3
    f1 = polynomValue(x1)
    f2 = polynomValue(x2)
    f3 = polynomValue(x3)
    new_f = min(f1, min(f2, f3))
    if new_f == f1:
        x = x1
        m = m1
        e = e1
    elif new_f == f2:
        x = x2
        m = m2
        e = e2
    elif new_f == f3:
        x = x3
        m = m3
        e = e3
    print("direction = \n", e)
    print("step = ", m)
    print("vector x = \n", x)
    print("polynom value = ", new_f, "\n")
    i = i + 1
