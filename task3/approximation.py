"""
    Аппроксимация функций
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


def func(x):
    # return x ** 2 + np.sin(x)
    return x ** 2 * np.cos(x)


def gaussFunc(a):  # метод Гаусса
    eps = 1e-16

    c = np.array(a)
    a = np.array(a)

    len1 = len(a[:, 0])
    len2 = len(a[0, :])
    vectB = copy.deepcopy(a[:, len1])

    for g in range(len1):

        max = abs(a[g][g])
        my = g
        t1 = g
        while t1 < len1:
            # for t1 in range(len(a[:,0])):
            if abs(a[t1][g]) > max:
                max = abs(a[t1][g])
                my = t1
            t1 += 1

        if abs(max) < eps:
            raise DetermExeption("Check determinant")

        if my != g:
            # a[g][:], a[my][:] = a[my][:], a[g][:]
            # numpy.swapaxes(a, 1, 0)
            b = copy.deepcopy(a[g])
            a[g] = copy.deepcopy(a[my])
            a[my] = copy.deepcopy(b)

        amain = float(a[g][g])

        z = g
        while z < len2:
            a[g][z] = a[g][z] / amain
            z += 1

        j = g + 1

        while j < len1:
            b = a[j][g]
            z = g

            while z < len2:
                a[j][z] = a[j][z] - a[g][z] * b
                z += 1
            j += 1

    a = backTrace(a, len1, len2)

    # print("Погрешность:")

    # print(vectorN(c, a, len1, vectB))

    return a


class DetermExeption(Exception):  # Ошибка, проверьте определитель матрицы
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def backTrace(a, len1, len2):
    a = np.array(a)
    i = len1 - 1
    while i > 0:
        j = i - 1

        while j >= 0:
            a[j][len1] = a[j][len1] - a[j][i] * a[i][len1]
            j -= 1
        i -= 1
    return a[:, len2 - 1]


def vectorN(c, a, len1, vectB):  # c-начальная матрица a-ответ len-ранг, vectB-вектор B
    c = np.array(c)
    a = np.array(a)
    vectB = np.array(vectB)

    b = np.zeros((len1))

    i = 0

    while i < len1:
        j = 0
        while j < len1:
            b[i] += c[i][j] * a[j]

            j += 1

        i = i + 1

    c = copy.deepcopy(b)
    print("!")

    for i in range(len1):
        c[i] = abs(c[i] - vectB[i])

    return c


def getX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i]
    return sum


def getXX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i] ** 2
    return sum


def getXXX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i] ** 3
    return sum


def getXXXX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i] ** 4
    return sum


def getXXXXX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i] ** 5
    return sum


def getXXXXXX(x, n):
    sum = 0
    for i in range(n):
        sum += x[i] ** 6
    return sum


def getY(y, n):
    sum = 0
    for i in range(n):
        sum += y[i]
    return sum


def getYX(y, n):
    sum = 0
    for i in range(n):
        sum += y[i] * x[i]
    return sum


def getYXX(y, n):
    sum = 0
    for i in range(n):
        sum += y[i] * x[i] * x[i]
    return sum


def getYXXX(y, n):
    sum = 0
    for i in range(n):
        sum += y[i] * x[i] * x[i] * x[i]
    return sum


def approximatefunc(coeff, x):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3]


def L0(x):
    return 1


def L1(x):
    return (-x)


def L2(x):
    return ((3 * x ** 2 - 1) / (2))


def L3(x):
    return ((3 * x - 5 * x ** 3) / (2))


def getQ(x, c0, c1, c2, c3):
    return c0 * L0(x) + c1 * L1(x) + c2 * L2(x) + c3 * L3(x)


# метод наименьших квадратов
n = int(input())
x = np.array([], dtype=float)
y = np.array([], dtype=float)
x = np.linspace(-1, 1, n)
y = x.copy()
for i in range(n):
    y[i] = func(x[i])
# x = np.linspace(-1, 1, n)
print(x)
print(y)

A = np.array([[getXXXXXX(x, n), getXXXXX(x, n), getXXXX(x, n), getXXX(x, n), getYXXX(y, n)],
              [getXXXXX(x, n), getXXXX(x, n), getXXX(x, n), getXX(x, n), getYXX(y, n)],
              [getXXXX(x, n), getXXX(x, n), getXX(x, n), getX(x, n), getYX(y, n)],
              [getXXX(x, n), getXX(x, n), getX(x, n), n, getY(y, n)]], dtype=float)

print(A)
# print(B)
coeff = gaussFunc(A)
xnew = np.linspace(np.min(x), np.max(x), 100)
ynew = [approximatefunc(coeff, xnew[i]) for i in range(100)]

# полиномы Лежандра
c0 = 2 * np.cos(1) - np.sin(1)
c1 = 0
c2 = 100 * np.sin(1) - 155 * np.cos(1)
c3 = 0
xnew2 = np.linspace(np.min(x), np.max(x), 100)
ynew2 = [getQ(xnew2[i], c0, c1, c2, c3) for i in range(100)]
# построение графиков
plt.plot(xnew, ynew, 'red')
plt.plot(xnew2, ynew2, 'green')
plt.plot(x, y, 'o')
plt.grid(True)
plt.show()
