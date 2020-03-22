import numpy as np
import matplotlib.pyplot as plt
import math

MAX_ITER = 1000
learning_rate = 10 ** -19


def count_mse(X, Y, W):
    return np.sum((X.dot(W) - Y) ** 2) / Y.size


def gradient_descent(X, Y, W, error_statistic):
    for _ in range(MAX_ITER):
        a = X.dot(W)
        loss = a - Y
        gradient = X.T.dot(loss)
        W = W - learning_rate * gradient
        cur_pred = count_nrmse_measure(X, Y, W)
        error_statistic.append(cur_pred)
    return W


def generalized_inverse(X, Y):
    generalized_inverse_matrix = np.linalg.pinv(X)
    return np.matmul(generalized_inverse_matrix, Y)


# Want to minimize this measure. Closer to 0 is better
def count_nrmse_measure(X, Y, W):
    return math.sqrt(np.sum((Y - (X.dot(W))) ** 2) / Y.size) / (np.max(Y) - np.min(Y))


def solve():
    for i in range(1, 8):
        with open(f'data/{i}.txt') as file:
            attr_num = int(file.readline())
            train_set_size = int(file.readline())
            X = []
            Y = []
            for j in range(0, train_set_size):
                object = [int(el) for el in file.readline().split()]
                X.append(object[:-1] + [1])
                Y.append(object[-1])
            X = np.array(X)
            Y = np.array(Y)
            gradient_W = np.zeros(X.shape[1])
            gradient_test_W = np.zeros(X.shape[1])
            errors_train_statistic = []
            gradient_W = gradient_descent(X, Y, gradient_W, errors_train_statistic)
            generalized_inverse_W = generalized_inverse(X, Y)
            test_set_size = int(file.readline())
            X_Test = []
            Y_Test = []
            for j in range(0, test_set_size):
                object = [int(el) for el in file.readline().split()]
                X_Test.append(object[:-1] + [1])
                Y_Test.append(object[-1])
            X_Test = np.array(X_Test)
            Y_Test = np.array(Y_Test)
            errors_test_statistic = []
            gradient_descent(X_Test, Y_Test, gradient_test_W, errors_test_statistic)
            print(f'Gradient descent NRMSE measure value {count_nrmse_measure(X_Test, Y_Test, gradient_W)}')
            print(
                f'Generalized inverse NRMSE measure value {count_nrmse_measure(X_Test, Y_Test, generalized_inverse_W)}')
            plt.plot(range(0, MAX_ITER), errors_train_statistic)
            plt.plot(range(0, MAX_ITER), errors_test_statistic)
            plt.ylabel('NRMSE')
            plt.xlabel('iter_num')
            plt.title(f'Dataset #{i}')
            plt.legend(('Train', 'Test'), loc='upper right')
            plt.show()


if __name__ == '__main__':
    solve()
