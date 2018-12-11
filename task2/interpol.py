"""
    Интерполирование функций
"""

import numpy as np
import matplotlib.pyplot as plt


# вычисление значения функции в точке
def func(x):
    y = x ** 2 + 4 * np.sin(x) - 2
    return y


# выбор узлов интерполирования
def xi(a, b, n, i):
    x = (1 / 2) * ((b - a) * np.cos(np.pi * (2 * i + 1) / (2 * (n + 1))) + (b + a))
    return x


def lagranz(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z


# строим полином Лагранжа для 3х узлов
# выбор узлов интерполирования
a = -np.pi / 2
b = np.pi / 2
n = 2
x = np.array([])
x = np.linspace(a, b, 3)
y = x
for i in range(n + 1):
    y[i] = func(x[i])
x = np.linspace(a, b, 3)
print(x)
print(y, '\n')
# построение полинома Лагранжа
xnew = np.linspace(np.min(x), np.max(x), 100)
ynew = [lagranz(x, y, i) for i in xnew]
plt.plot(xnew, xnew ** 2 + 4 * np.sin(xnew) - 2, 'green')  # построение изначального графика
plt.plot(x, y, 'o', xnew, ynew, 'blue')  # построение полинома лагранжа
plt.grid(True)
plt.show()
# построение графика погрешностей
x = np.linspace(np.min(x), np.max(x), 100)
y = x
for i in range(100):
    y[i] = func(x[i])
x = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x, abs(ynew - y), 'red')
plt.grid(True)
plt.show()
# строим полином Лагранжа для 3х узлов с выбором узлов интерполирования по формуле 3.2
x = np.linspace(a, b, 3)
y = x
for i in range(n + 1):
    x[i] = xi(a, b, n, i)
    y[i] = func(x[i])
x = np.array([])
x = np.linspace(a, b, 3)
for i in range(n + 1):
    x[i] = xi(a, b, n, i)
x[1] = 0
print(x)
print(y, '\n')
# построение полинома Лагранжа
xnew = np.linspace(np.min(x), np.max(x), 100)
ynew = [lagranz(x, y, i) for i in xnew]
plt.plot(xnew, xnew ** 2 + 4 * np.sin(xnew) - 2, 'green')  # построение изначального графика
plt.plot(x, y, 'o', xnew, ynew, 'blue')  # построение полинома лагранжа
plt.grid(True)
plt.show()
# построение графика погрешностей
x = np.linspace(np.min(x), np.max(x), 100)
y = x
for i in range(100):
    y[i] = func(x[i])
x = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x, abs(ynew - y), 'red')
plt.grid(True)
plt.show()

# строим полином Лагранжа для 4ех узлов
# выбор узлов интерполирования
a = -np.pi / 2
b = np.pi / 2
n = 3
x = np.array([])
x = np.linspace(a, b, 4)
y = x
for i in range(n + 1):
    y[i] = func(x[i])
x = np.linspace(a, b, 4)
print(x)
print(y, '\n')
# построение полинома Лагранжа
xnew = np.linspace(np.min(x), np.max(x), 100)
ynew = [lagranz(x, y, i) for i in xnew]
plt.plot(xnew, xnew ** 2 + 4 * np.sin(xnew) - 2, 'green')  # построение изначального графика
plt.plot(x, y, 'o', xnew, ynew, 'blue')  # построение полинома лагранжа
plt.grid(True)
plt.show()
# построение графика погрешностей
x = np.linspace(np.min(x), np.max(x), 100)
y = x
for i in range(100):
    y[i] = func(x[i])
x = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x, abs(ynew - y), 'red')
plt.grid(True)
plt.show()
# строим полином Лагранжа для 4ех узлов с выбором узлов интерполирования по формуле 3.2
x = np.linspace(a, b, 4)
y = x
for i in range(n + 1):
    x[i] = xi(a, b, n, i)
    y[i] = func(x[i])
x = np.array([])
x = np.linspace(a, b, 4)
for i in range(n + 1):
    x[i] = xi(a, b, n, i)
print(x)
print(y, '\n')
# построение полинома Лагранжа
xnew = np.linspace(np.min(x), np.max(x), 100)
ynew = [lagranz(x, y, i) for i in xnew]
plt.plot(xnew, xnew ** 2 + 4 * np.sin(xnew) - 2, 'green')  # построение изначального графика
plt.plot(x, y, 'o', xnew, ynew, 'blue')  # построение полинома лагранжа
plt.grid(True)
plt.show()
# построение графика погрешностей
x = np.linspace(np.min(x), np.max(x), 100)
y = x
for i in range(100):
    y[i] = func(x[i])
x = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x, abs(ynew - y), 'red')
plt.grid(True)
plt.show()
