"""
    Метод наискорейшего градиентного спуска
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


def gradientDescent(A, x, b):
    """
    Вычисление градиента
    :param A: матрица А
    :param x: вектор х
    :param b: вектор b
    :return: градиент
    """
    return np.dot(A, x) + b


def stepValue(q, A):
    """
    Вычисление шага m
    :param q: градиент
    :param A: матрица А
    :return: шаг m
    """
    q_transpose = q.transpose()
    numerator = np.dot(q_transpose, q)
    denominator = np.dot(q_transpose, np.dot(A, q))
    return -numerator / denominator


# constants
A = np.array([[4, 1, 1], [1, 6.2, -1], [1, -1, 8.2]])
b = np.array([[1], [-2], [3]])
epsilon = 0.0000001
# first member
x = np.array([[1], [0], [0]])
new_f = polynomValue(x)
f = 0
print("iteration: 1")
print("vector x = \n", x)
print("polynom value = ", new_f, "\n")
i = 2
while abs(f - new_f) > epsilon:
    print("iteration: ", i)
    f = new_f
    q = gradientDescent(A, x, b)
    m = stepValue(q, A)
    x = x + m * q
    new_f = polynomValue(x)
    print("gradient = \n", q)
    print("step = ", m)
    print("vector x = \n", x)
    print("polynom value = ", new_f, "\n")
    i = i + 1
