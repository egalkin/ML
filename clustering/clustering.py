import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA
from datetime import datetime


def dbscan(unlabeled_data, eps, min_pts):
    labels = [0] * len(unlabeled_data)
    labels_distribution = {}
    current_cluster_id = 0

    for object_id in range(0, len(unlabeled_data)):
        if labels[object_id] != 0:
            continue

        neighbours = count_neighbours(unlabeled_data, object_id, eps)

        if len(neighbours) < min_pts:
            labels[object_id] = -1
        else:
            current_cluster_id += 1
            build_claster(unlabeled_data, labels, labels_distribution, object_id, neighbours, current_cluster_id, eps,
                          min_pts)

    labels_distribution[-1] = []
    for i in range(0, len(labels)):
        if labels[i] == -1:
            labels_distribution[-1].append(i)
    return labels, labels_distribution


def count_neighbours(unlabeled_data, object_id, eps):
    neighbours = []
    min_dist = 100000.0
    max_dist = 0
    mean_dist = 0

    mean_dist = 0.0
    for cur_object_id in range(0, len(unlabeled_data)):
        if object_id == cur_object_id:
            continue
        distance = np.linalg.norm(unlabeled_data[object_id] - unlabeled_data[cur_object_id])
        mean_dist += distance
        if distance <= eps:
            neighbours.append(cur_object_id)
        min_dist = min(min_dist, distance)
        max_dist = max(max_dist, distance)
        mean_dist += distance
    mean_dist = mean_dist / len(unlabeled_data)
    # print(min_dist, max_dist, mean_dist)
    return neighbours


def build_claster(unlabeled_data, labels, labels_distribution, object_id, neighbours, current_cluster_id, eps, min_pts):
    labels[object_id] = current_cluster_id
    if current_cluster_id not in labels_distribution:
        labels_distribution[current_cluster_id] = [object_id]
    else:
        labels_distribution[current_cluster_id].append(object_id)

    i = 0
    while i < len(neighbours):
        neighbour = neighbours[i]
        if labels[neighbour] <= 0:
            labels_distribution[current_cluster_id].append(neighbour)
        if labels[neighbour] == -1:
            labels[neighbour] = current_cluster_id
        elif labels[neighbour] == 0:
            labels[neighbour] = current_cluster_id
            snake_neighbours = count_neighbours(unlabeled_data, neighbour, eps)
            if len(snake_neighbours) >= min_pts:
                neighbours += snake_neighbours
        i += 1


def preprocess(classes_distribution, data_set):
    for i in range(len(data_set)):
        cls = data_set.iloc[i]['class']
        if cls in classes_distribution:
            classes_distribution[cls].append(i)
        else:
            classes_distribution[cls] = [i]


def count_measure(unlabeled_data, labels):
    mean_inner_distance = count_mean_inner_distance(unlabeled_data, labels)
    mean_outer_distance = count_mean_outer_distance(unlabeled_data, labels)
    if mean_outer_distance == 0 or mean_inner_distance == 0:
        return 10000000.0
    return mean_inner_distance / mean_outer_distance


def count_mean_inner_distance(unlabeled_data, labels):
    numerator, denominator = 0.0, 1
    for i in range(0, len(unlabeled_data)):
        for j in range(0, len(unlabeled_data)):
                if i < j and labels[i] == labels[j]:
                    numerator += np.linalg.norm(unlabeled_data[i] - unlabeled_data[j])
                    denominator += 1
    return numerator / denominator


def count_mean_outer_distance(unlabeled_data, labels):
    numerator, denominator = 0.0, 1
    for i in range(0, len(unlabeled_data)):
        for j in range(0, len(unlabeled_data)):
                if i < j and labels[i] != labels[j]:
                    numerator += np.linalg.norm(unlabeled_data[i] - unlabeled_data[j])
                    denominator += 1
    return numerator / denominator


# mean_dist = [0.02, 0.1]
# 0.16820071218859642 0.006 6
# 0.21209725113248878 0.005 3 6
# 0.2368541523014908 0.006 3 7
def count_optimal_clastering_parameters(unlabeled_data):
    measure = 10000000.0
    noise = 1000000
    optimal_eps = 0
    optimal_min_pts = 0
    for eps in np.arange(0.020, 0.05, 0.001):
        for min_pts in range(1, 10):
            labels, labels_distribution = dbscan(unlabeled_data, eps, min_pts)
            cur_measure = count_measure(unlabeled_data, labels)
            cur_noise = len(labels_distribution[-1])
            if len(labels_distribution.keys()) == 4:
                print(cur_measure, eps, min_pts, len(labels_distribution.keys()), len(labels_distribution[-1]))
            if measure != 10000000.0 and cur_measure == 10000000.0:
                break
            if cur_measure < measure and len(labels_distribution.keys()) == 4:
                measure = cur_measure
                noise = cur_noise
                optimal_eps = eps
                optimal_min_pts = min_pts
                print(measure, optimal_eps, optimal_min_pts, len(labels_distribution.keys()), noise, "||")
    return measure, optimal_eps, optimal_min_pts


# 0.29996778588510786 0.02 19

def solve():
    data_set = pd.read_csv('dataset_61_iris.csv')
    classes_distribution = {}
    preprocess(classes_distribution, data_set)
    classes_to_numbers = {}
    for k, v in classes_distribution.items():
        print(k, ":", len(v))
    for idx, cl in enumerate(classes_distribution.keys()):
        classes_to_numbers[cl] = idx + 1
    unlabeled_data = preprocessing.normalize(data_set.values[:, :-1])
    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(unlabeled_data))
    y = np.array(list(map(lambda x: classes_to_numbers[x], data_set.values[:, -1])))

    plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Iris-setosa', c='red')
    plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Iris-versicolor', c='blue')
    plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Iris-virginica', c='lightgreen')

    plt.legend()
    plt.show()

    # measure, optimal_eps, optimal_min_pts = count_optimal_clastering_parameters(unlabeled_data)
    # print(measure, optimal_eps, optimal_min_pts)
    labels, labels_distribution = dbscan(unlabeled_data, 0.0258, 2)
    measure = count_measure(unlabeled_data, labels)
    print(measure)
    labels = np.array(labels)

    plt.scatter(transformed[labels == 1][0], transformed[labels == 1][1], label='Iris-setosa', c='red')
    plt.scatter(transformed[labels == 2][0], transformed[labels == 2][1], label='Iris-versicolor', c='blue')
    plt.scatter(transformed[labels == 3][0], transformed[labels == 3][1], label='3', c='pink')
    plt.scatter(transformed[labels == 4][0], transformed[labels == 4][1], label='Iris-virginica', c='lightgreen')
    plt.scatter(transformed[labels == 5][0], transformed[labels == 5][1], label='5', c='yellow')
    plt.scatter(transformed[labels == -1][0], transformed[labels == -1][1], label='Noise', c='black')
    plt.legend()
    plt.show()
    # for k, v in labels_distribution.items():
    #     print(k, ":", len(v), ";", v)

# 0.8953942474551123 0.02 5 4 109
# 0.8953942474551123 0.02 5 4 109 ||
# 0.9141622475162104 0.02 6 4 110
# 0.8776898000712366 0.021 6 4 108
# 0.8776898000712366 0.021 6 4 108 ||
# 0.6688838076381247 0.022000000000000002 5 4 82
# 0.6688838076381247 0.022000000000000002 5 4 82 ||
# 0.8956948335599092 0.022000000000000002 7 4 106
# 0.7140666910396873 0.023000000000000003 7 4 95
# 0.7129235598909199 0.024000000000000004 7 4 82
# 0.6709594371786471 0.025000000000000005 7 4 78
# 0.8448461326432458 0.025000000000000005 9 4 99
# 0.29603640665981423 0.026000000000000006 3 4 32
# 0.29603640665981423 0.026000000000000006 3 4 32 ||
# 0.6546134387552515 0.026000000000000006 8 4 77
# 0.8214485177539043 0.026000000000000006 9 4 95
# 0.24539397680407068 0.027000000000000007 2 4 23
# 0.24539397680407068 0.027000000000000007 2 4 23 ||
# 0.27125492139687946 0.027000000000000007 3 4 28
# 0.4037622207371483 0.027000000000000007 6 4 43
# 0.24089764906992026 0.028000000000000008 2 4 22
# 0.24089764906992026 0.028000000000000008 2 4 22 ||
# 0.24865874053721862 0.028000000000000008 3 4 24
# 0.27103094130974403 0.028000000000000008 4 4 28
# 0.28846424785916924 0.028000000000000008 5 4 31
# 0.3618235230751232 0.028000000000000008 6 4 40
# 0.2276027075157005 0.02900000000000001 2 4 19
# 0.2276027075157005 0.02900000000000001 2 4 19 ||
# 0.2360842195346694 0.02900000000000001 3 4 21
# 0.26094324668054536 0.02900000000000001 4 4 26
# 0.2721953117380147 0.02900000000000001 5 4 28
# 0.34775179245929133 0.02900000000000001 6 4 38
# 0.36901325869313695 0.02900000000000001 7 4 41
# 0.4299540480617555 0.02900000000000001 8 4 48
# 0.5232107833076892 0.02900000000000001 9 4 62
# 0.2276027075157005 0.03000000000000001 2 4 19
# 0.2360842195346694 0.03000000000000001 3 4 21
# 0.2513454359018256 0.03000000000000001 4 4 24
# 0.26648585778714606 0.03000000000000001 5 4 27
# 0.31406770087344427 0.03000000000000001 7 4 35
# 0.3551539965303812 0.03000000000000001 8 4 39
# 0.4364421583820796 0.03000000000000001 9 4 51
# 0.22163429634348908 0.03100000000000001 3 4 17
# 0.22163429634348908 0.03100000000000001 3 4 17 ||
# 0.23874341024325882 0.03100000000000001 4 4 21
# 0.23874341024325882 0.03100000000000001 5 4 21
# 0.310951700020941 0.03100000000000001 8 4 34
# 0.3871402125271422 0.03100000000000001 9 4 42
# 0.22163429634348908 0.032000000000000015 3 4 17
# 0.2305637926085865 0.032000000000000015 4 4 19
# 0.23500997637647666 0.032000000000000015 5 4 20
# 0.34097114140764345 0.032000000000000015 9 4 36
# 0.20677784327248053 0.033000000000000015 1 4 11
# 0.20677784327248053 0.033000000000000015 1 4 11 ||
# 0.20677784327248053 0.033000000000000015 2 4 11
# 0.2205862030584904 0.033000000000000015 4 4 16
# 0.22865120548937776 0.033000000000000015 5 4 18
# 0.3117715649423397 0.033000000000000015 9 4 33
# 0.20456746366763479 0.034000000000000016 1 4 10
# 0.20456746366763479 0.034000000000000016 1 4 10 ||
# 0.20456746366763479 0.034000000000000016 2 4 10
# 0.20456746366763479 0.034000000000000016 3 4 10
# 0.21780502153758438 0.034000000000000016 4 4 15
# 0.21780502153758438 0.034000000000000016 5 4 15
# 0.21327391789262687 0.03500000000000002 4 4 13
# 0.21327391789262687 0.03500000000000002 5 4 13
# 0.2169347472865293 0.03500000000000002 6 4 14
# 0.20456746366763479 0.034000000000000016 1

if __name__ == '__main__':
    solve()
