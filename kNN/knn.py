import numpy as np
import pandas as pd
import nonparametric_regression as nr
import matplotlib.pyplot as plt
import f_measure as fm
from sklearn import preprocessing
from datetime import datetime

STEP_DIVIDER = 500.0


def preprocess(classes_distribution, data_set):
    for i in range(len(data_set)):
        cls = data_set.iloc[i]['Type']
        if cls in classes_distribution:
            classes_distribution[cls].append(i)
        else:
            classes_distribution[cls] = [i]


def parzen_window(X, excluded_idx, classes_distribution, classes_to_numbers, q, distance_func, kernel_func, h):
    classes_results = [0] * len(classes_distribution.keys())
    for cls in classes_distribution.keys():
        cls_counted_function = 0
        for i in classes_distribution[cls]:
            if i != excluded_idx:
                cls_counted_function += kernel_func(nr.div(distance_func(X[i], q), h))
        classes_results[classes_to_numbers[cls]] = cls_counted_function
    return classes_results.index(max(classes_results))


def count_distances(distances, X, distance_func):
    max_dist = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                cur_dist = distance_func(X[i], X[j])
                distances[i].append(cur_dist)
                max_dist = max(cur_dist, max_dist)
        distances[i].sort()
    return max_dist


def test_parameters(X, Y, nearest_neighbours, num_of_classes, classes_distribution, classes_to_numbers,
                    distance_func,
                    kernel_func, k, is_fixed_window):
    confusion_matrix = np.zeros((num_of_classes, num_of_classes), dtype=np.int)
    for i in range(len(X)):
        xi = X[i]
        yi = classes_to_numbers[Y[i]]
        h = k if is_fixed_window else nearest_neighbours[i][k]
        ai = parzen_window(X, i, classes_distribution, classes_to_numbers, xi, distance_func,
                           kernel_func, h)
        confusion_matrix[ai][yi] += 1
    return fm.count_micro_f_measure(confusion_matrix, num_of_classes)


def window_regression(X, Y, classes_distribution, is_fixed_window):
    distances = nr.Distances.__dict__.copy()
    kernels = nr.Kernels.__dict__.copy()
    removed_fields = ['__module__', '__dict__', '__weakref__', '__doc__']
    for field in removed_fields:
        del distances[field]
        del kernels[field]
    del distances['minkowski']
    num_of_classes = len(classes_distribution.keys())
    classes_to_numbers = {}
    for idx, cl in enumerate(classes_distribution.keys()):
        classes_to_numbers[cl] = idx
    f_measure = 0
    best_values = (None, None, 0, 0)
    for distance_func_name in distances.keys():
        distance_func = distances[distance_func_name].__func__
        nearest_neighbours = [[] for i in range(len(X))]
        max_dist = count_distances(nearest_neighbours, X, distance_func)
        windows_step = max_dist / STEP_DIVIDER
        window_range = np.arange(windows_step, max_dist, windows_step) if is_fixed_window else range(len(X) - 1)
        for kernel_func_name in kernels.keys():
            kernel_func = kernels[kernel_func_name].__func__
            for k in window_range:
                cur_f_measure = test_parameters(X, Y, nearest_neighbours, num_of_classes, classes_distribution,
                                                classes_to_numbers,
                                                distance_func, kernel_func, k, is_fixed_window)
                if cur_f_measure > f_measure:
                    f_measure = cur_f_measure
                    best_values = (distance_func_name, kernel_func_name, k, cur_f_measure)
    return best_values


# ('manhattan', 'logistic', 0.0018557025517033248, 0.7646143821746378) - fixed
# ('manhattan', 'quartic', 9, 0.7694711581915796) - variable

def build_graphics(X, Y, classes_distribution, distance_name, kernel_name, is_fixed_window):
    distance_func = nr.Distances.__dict__[distance_name].__func__
    kernel_func = nr.Kernels.__dict__[kernel_name].__func__
    num_of_classes = len(classes_distribution.keys())
    classes_to_numbers = {}
    for idx, cl in enumerate(classes_distribution.keys()):
        classes_to_numbers[cl] = idx
    nearest_neighbours = [[] for i in range(len(X))]
    max_dist = count_distances(nearest_neighbours, X, distance_func)
    windows_step = max_dist / STEP_DIVIDER
    window_range = np.arange(windows_step, max_dist, windows_step) if is_fixed_window else range(len(X) - 1)
    f_measure_values = []
    for k in window_range:
        f_measure = test_parameters(X, Y, nearest_neighbours, num_of_classes, classes_distribution,
                                    classes_to_numbers,
                                    distance_func, kernel_func, k, is_fixed_window)
        f_measure_values.append(f_measure)
    plt.plot(window_range, f_measure_values)
    plt.title("dist_name: '{}' kernel_name: '{}'".format(distance_name, kernel_name))
    plt.ylabel('f measure')
    plt.xlabel('window steps')
    plt.show()


def solve():
    data_set = pd.read_csv('dataset_41_glass.csv')
    classes_distribution = {}
    preprocess(classes_distribution, data_set)
    X = data_set.values[:, :-1]
    Y = data_set.values[:, -1]
    normalized_X = preprocessing.normalize(X)
    # build_graphics(normalized_X, Y, classes_distribution, 'manhattan', 'logistic', True)
    build_graphics(normalized_X, Y, classes_distribution, 'manhattan', 'quartic', False)
    # print(datetime.now())
    # print(window_regression(normalized_X, Y, classes_distribution, True))
    # print(datetime.now())
    print(datetime.now())
    print(window_regression(normalized_X, Y, classes_distribution, False))
    print(datetime.now())


if __name__ == '__main__':
    solve()
