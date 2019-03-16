import numpy as np
from sklearn.datasets import fetch_mldata
from heapq import *
from scipy.spatial import distance
import matplotlib.pyplot as plt


def process_data(data, query, labels):
    h = []
    for i in range(len(data)):
        heappush(h, (distance.euclidean(data[i], query), labels[i]))
    return h


def get_label(processed_data, k):
    d = {}
    for i in range(k):
        nn = heappop(processed_data)[1]
        d[nn] = 1 if nn not in d else d[nn] + 1
    return max(d, key=d.get)


def knn(data, labels, query, k):
    processed_data = process_data(data, query, labels)
    return get_label(processed_data, k)


def q1b(train, train_labels, test, test_labels):
    n = 1000
    k = 10
    accuracy = 1
    for i in range(len(test)):
        if knn(train[:n], train_labels[:n], test[i], k) != test_labels[i]:
            accuracy -= (1.0 / len(test))
    print "accuracy:", accuracy


def q1c(train, train_labels, test, test_labels):
    n = 1000
    k_max = 100
    accuracies = np.ones(k_max)
    for i in range(len(test)):
        processed_data = process_data(train[:n], test[i], train_labels[:n])
        for k in range(k_max):
            learned_label = get_label(processed_data[:], k + 1)
            if learned_label != test_labels[i]:
                accuracies[k] -= (1.0 / n)
    plt.plot([k + 1 for k in range(k_max)], accuracies, '.')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.axis([-1, 105, 0.65, 1.0])
    plt.xticks(np.arange(0, 105, 5))
    plt.show()



def q1d(train, train_labels, test, test_labels):
    k = 1
    n_min = 100
    n_max = 5000
    step = 100
    accuracies = np.ones(n_max / step)
    for i in range(len(test)):
        if i % 100 == 0:
            print "Processing: ", 100 * float(i) / len(test), "%"
        processed_data = []
        for n in range(n_min, n_max + 1, step):
            processed_data = processed_data + process_data(train[n-step:n], test[i], train_labels[n-step:n])
            heapify(processed_data)
            if get_label(processed_data[:], k) != test_labels[i]:
                accuracies[(n / step) - 1] -= (1.0 / len(test))
    plt.plot([n for n in range(n_min, n_max + 1, step)], accuracies, '.')
    plt.grid()
    plt.xlabel('n')
    plt.ylabel('accuracy')
    plt.axis([0, 5100, 0.6, 1.0])
    plt.xticks(np.arange(0, 5000, 500))
    plt.yticks(np.arange(0.6, 1.05, 0.05))
    plt.show()


if __name__ == "__main__":
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    q1b(train, train_labels, test, test_labels)
    q1c(train, train_labels, test, test_labels)
    q1d(train, train_labels, test, test_labels)


