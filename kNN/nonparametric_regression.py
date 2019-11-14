import math
import pandas as pd


class Distances:

    @staticmethod
    def minkowski(x, y, p):
        dist = 0
        for i in range(len(x)):
            dist += abs(x[i] - y[i]) ** p
        return dist ** (1 / p)

    @staticmethod
    def manhattan(x, y):
        return Distances.minkowski(x, y, 1)

    @staticmethod
    def euclidean(x, y):
        return Distances.minkowski(x, y, 2)

    @staticmethod
    def chebyshev(x, y):
        dist = 0
        for i in range(len(x)):
            dist = max(dist, abs(x[i] - y[i]))
        return dist


class Kernels:
    @staticmethod
    def uniform(u):
        return 1 / 2 if abs(u) < 1 else 0.0

    @staticmethod
    def triangular(u):
        return 1 - abs(u) if abs(u) < 1 else 0.0

    @staticmethod
    def epanechnikov(u):
        return 3 / 4 * (1 - u ** 2) if abs(u) < 1 else 0.0

    @staticmethod
    def quartic(u):
        return 15 / 16 * (1 - u ** 2) ** 2 if abs(u) < 1 else 0.0

    @staticmethod
    def triweight(u):
        return 35 / 32 * (1 - u ** 2) ** 3 if abs(u) < 1 else 0.0

    @staticmethod
    def tricube(u):
        return 70 / 81 * (1 - abs(u) ** 3) ** 3 if abs(u) < 1 else 0.0

    @staticmethod
    def gaussian(u):
        return ((2 * math.pi) ** (-1 / 2)) * math.e ** (-1 / 2 * u ** 2)

    @staticmethod
    def cosine(u):
        return math.pi / 4 * math.cos(math.pi / 2 * u) if abs(u) < 1 else 0.0

    @staticmethod
    def logistic(u):
        return 1 / (math.e ** u + 2 + math.e ** -u)

    @staticmethod
    def sigmoid(u):
        return 2 / math.pi * (1 / (math.e ** u + math.e ** -u))


def div(a, b):
    return 1 if b == 0 else a / b


def kernel_smoothing(n, feature_matrix, q, distance_func, kernel_func, h):
    numerator = 0
    denominator = 0
    for i in range(n):
        if h == 0:
            weight = kernel_func(0) if feature_matrix[i][:-1] == q else 0
        else:
            weight = kernel_func(div(distance_func(feature_matrix[i][:-1], q), h))
        numerator += feature_matrix[i][-1] * weight
        denominator += weight
    return numerator, denominator


def count_neighbour(feature_matrix, q, distance_func, window_parameter):
    neighbours = []
    for i in range(len(feature_matrix)):
        neighbours.append(distance_func(q, feature_matrix.iloc[i][:-1]))
    return sorted(neighbours)[window_parameter - 1]


def solve():
    n, m = tuple(map(int, input().split()))
    feature_matrix = []
    func_values = {}
    default_value = 0
    for i in range(n):
        object_description = list(map(int, input().split()))
        func_values[tuple(object_description[:-1])] = object_description[-1]
        default_value += object_description[-1]
        feature_matrix.append(object_description)
    q = list(map(int, input().split()))
    distance_func = Distances.__dict__[input()].__func__
    kernel_func = Kernels.__dict__[input()].__func__
    window = input()
    window_parameter = int(input())
    h = window_parameter if window == 'fixed' else count_neighbour(feature_matrix, q, distance_func,
                                                                   window_parameter + 1)
    numerator, denominator = kernel_smoothing(n, feature_matrix, q, distance_func, kernel_func, h)
    if numerator == 0 and denominator == 0:
        print(default_value / n)
    else:
        print(div(numerator, denominator))


if __name__ == '__main__':
    solve()
