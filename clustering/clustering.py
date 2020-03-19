import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA

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


def dbscan(unlabeled_data, eps, min_pts):
    labels = [0] * len(unlabeled_data)
    labels_distribution = {}
    current_cluster_id = 0

    for object_id in range(0, len(unlabeled_data)):
        if labels[object_id] == 0:

            neighbours = count_neighbours(unlabeled_data, object_id, eps)

            if len(neighbours) < min_pts:
                labels[object_id] = -1
            else:
                current_cluster_id += 1
                expand(unlabeled_data, labels, labels_distribution, object_id, neighbours, current_cluster_id,
                       eps,
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
        distance = Distances.euclidean(unlabeled_data[object_id], unlabeled_data[cur_object_id])
        mean_dist += distance
        if distance <= eps:
            neighbours.append(cur_object_id)
        min_dist = min(min_dist, distance)
        max_dist = max(max_dist, distance)
        mean_dist += distance
    mean_dist = mean_dist / len(unlabeled_data)
    # print(min_dist, max_dist, mean_dist)
    return neighbours


def expand(unlabeled_data, labels, labels_distribution, object_id, neighbours, current_cluster_id, eps, min_pts):
    labels[object_id] = current_cluster_id
    if current_cluster_id not in labels_distribution:
        labels_distribution[current_cluster_id] = [object_id]
    else:
        labels_distribution[current_cluster_id].append(object_id)

    for neighbour in neighbours:
        if labels[neighbour] == 0:
            labels[neighbour] = current_cluster_id
            snake_neighbours = count_neighbours(unlabeled_data, neighbour, eps)
            if len(snake_neighbours) >= min_pts:
                neighbours += snake_neighbours


def preprocess(classes_distribution, data_set):
    for i in range(len(data_set)):
        cls = data_set.iloc[i]['class']
        if cls in classes_distribution:
            classes_distribution[cls].append(i)
        else:
            classes_distribution[cls] = [i]


def count_inner_measure(unlabeled_data, labels):
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


# Rand measure = (TP + TN) / (TP + TN + FP + FN) == Accuracy
def count_outer_measure(unlabeled_data, true_labels, labels):
    TP = 0
    TN = 0
    for i in range(0, len(unlabeled_data)):
        for j in range(0, len(unlabeled_data)):
            if (i < j):
                if true_labels[i] == true_labels[j] and labels[i] == labels[j]:
                    TP += 1
                if true_labels[i] != true_labels[j] and labels[i] != labels[j]:
                    TN += 1
    return (TP + TN + 0.0) / len(unlabeled_data)


class ClusteringParams:
    def __init__(self, measure, optimal_eps, optimal_min_pts, eps_values, min_pts_values, measure_values):
        self.measure = measure
        self.optimal_eps = optimal_eps
        self.optimal_min_pts = optimal_min_pts
        self.eps_values = eps_values
        self.min_pts_values = min_pts_values
        self.measure_values = measure_values


def count_optimal_clustering_parameters(unlabeled_data, true_labels, use_outer_measure=True):
    measure = 0.0 if use_outer_measure else 10000000.0
    optimal_eps = 0
    optimal_min_pts = 0
    eps_values = []
    min_pts_values = []
    measure_values = []
    for eps in np.arange(0.001, 0.1, 0.001):
        for min_pts in range(1, 50):
            labels, labels_distribution = dbscan(unlabeled_data, eps, min_pts)
            cur_measure = count_outer_measure(unlabeled_data, true_labels,
                                              labels) if use_outer_measure else count_inner_measure(unlabeled_data,
                                                                                                    labels)
            if cur_measure != 10000000.0:
                eps_values.append(eps)
                min_pts_values.append(min_pts)
                measure_values.append(cur_measure)
            if (cur_measure < measure and not use_outer_measure) or (cur_measure > measure and use_outer_measure):
                measure = cur_measure
                optimal_eps = eps
                optimal_min_pts = min_pts
    return ClusteringParams(measure, optimal_eps, optimal_min_pts, eps_values, min_pts_values, measure_values)


def solve():
    data_set = pd.read_csv('dataset_61_iris.csv')
    classes_distribution = {}
    preprocess(classes_distribution, data_set)
    classes_to_numbers = {}
    for idx, cl in enumerate(classes_distribution.keys()):
        classes_to_numbers[cl] = idx + 1
    unlabeled_data = preprocessing.normalize(data_set.values[:, :-1])
    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(unlabeled_data))
    true_labels = np.array(list(map(lambda x: classes_to_numbers[x], data_set.values[:, -1])))

    plt.scatter(transformed[true_labels == 1][0], transformed[true_labels == 1][1], label='Iris-setosa', c='red')
    plt.scatter(transformed[true_labels == 2][0], transformed[true_labels == 2][1], label='Iris-versicolor', c='blue')
    plt.scatter(transformed[true_labels == 3][0], transformed[true_labels == 3][1], label='Iris-virginica',
                c='lightgreen')

    plt.title("True data distribution")
    plt.legend()
    plt.show()

    # best params for clustering with outer measure: eps = 0.046, min_pts = 24
    clustering_params = count_optimal_clustering_parameters(unlabeled_data, true_labels)
    print(clustering_params.measure, clustering_params.optimal_eps, clustering_params.optimal_min_pts)
    labels, labels_distribution = dbscan(unlabeled_data, clustering_params.optimal_eps,
                                         clustering_params.optimal_min_pts)

    ax = plt.axes(projection='3d')

    eps_values = np.array(clustering_params.eps_values)
    min_pts_values = np.array(clustering_params.min_pts_values)
    measure_values = np.array(clustering_params.measure_values)
    ax.scatter3D(eps_values, min_pts_values, measure_values, c=measure_values, cmap='Greens')
    ax.set_xlabel('eps')
    ax.set_ylabel('min_pts')
    plt.title("Outer measure based clustering")
    plt.show()

    labels = np.array(labels)
    plt.scatter(transformed[labels == 1][0], transformed[labels == 1][1], label='Cluster 1', c='red')
    plt.scatter(transformed[labels == 2][0], transformed[labels == 2][1], label='Cluster 2', c='blue')
    plt.scatter(transformed[labels == 3][0], transformed[labels == 3][1], label='Cluster 3', c='lightgreen')
    plt.scatter(transformed[labels == -1][0], transformed[labels == -1][1], label='Noise', c='black')
    plt.title("Outer measure based clustering")
    plt.legend()
    plt.show()

    # # best params for clustering with outer measure: eps = 0.072, min_pts = 1
    clustering_params = count_optimal_clustering_parameters(unlabeled_data, true_labels, False)
    print(clustering_params.measure, clustering_params.optimal_eps, clustering_params.optimal_min_pts)
    labels, labels_distribution = dbscan(unlabeled_data, clustering_params.optimal_eps,
                                         clustering_params.optimal_min_pts)

    ax = plt.axes(projection='3d')

    eps_values = np.array(clustering_params.eps_values)
    min_pts_values = np.array(clustering_params.min_pts_values)
    measure_values = np.array(clustering_params.measure_values)
    ax.scatter3D(eps_values, min_pts_values, measure_values, c=measure_values, cmap='Greens')
    ax.set_xlabel('eps')
    ax.set_ylabel('min_pts')
    plt.title("Inner measure based clustering")
    plt.show()

    labels = np.array(labels)
    plt.scatter(transformed[labels == 1][0], transformed[labels == 1][1], label='Cluster 1', c='red')
    plt.scatter(transformed[labels == 2][0], transformed[labels == 2][1], label='Cluster 2', c='blue')
    plt.scatter(transformed[labels == -1][0], transformed[labels == -1][1], label='Noise', c='black')
    plt.title("Inner measure based clustering")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    solve()
