#################################
# Your name: Raz Landau
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.datasets import fetch_mldata
"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Pre processing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    data = sklearn.preprocessing.normalize(data)
    w = np.zeros(len(data[0]))
    for sample, label in zip(data, labels):
        w += (get_prediction(w, sample) != label) * sample * label
    return w


#################################

# Place for additional code

def get_prediction(w, sample):
    return 1 if (np.dot(w, sample)) >= 0 else -1


def get_accuracy(w, data, labels):
    hits = 0.0
    for sample, label in zip(data, labels):
        hits += get_prediction(w, sample) == label
    return hits / len(data)


def q1a(train_data, train_labels, test_data, test_labels):
    n_values = [5, 10, 50, 100, 500, 1000, 5000]
    runs = 100
    means, bottom_percentiles, top_percentiles = [], [], []
    for n in n_values:
        print("n =", n)
        accuracies = []
        for _ in np.arange(runs):
            permutation = np.random.permutation(n)
            w = perceptron(
                [train_data[:n][i] for i in permutation],
                [train_labels[:n][i] for i in permutation],
            )
            accuracies.append(get_accuracy(w, test_data, test_labels))
        means.append(np.mean(accuracies))
        bottom_percentiles.append(np.percentile(accuracies, 5))
        top_percentiles.append(np.percentile(accuracies, 95))

    titles = ["n", "mean", "5%", "95%"]
    data = zip(n_values, means, bottom_percentiles, top_percentiles)
    row_format = "{:>20}" * (len(titles))
    print(row_format.format(*titles))
    for row in data:
        print(row_format.format(*row))


def q1bcd(train_data, train_labels, test_data, test_labels):
    w = perceptron(train_data, train_labels)

    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.savefig('q1b.png')

    print("q1c: ", get_accuracy(w, test_data, test_labels))

    for test, label in zip(test_data, test_labels):
        prediction = 1 if (np.dot(w, test)) >= 0 else -1
        if prediction != label:
            plt.imshow(np.reshape(test, (28, 28)), interpolation='nearest')
            plt.savefig("q1d.png")
            break


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    q1a(train_data, train_labels, test_data, test_labels)
    q1bcd(train_data, train_labels, test_data, test_labels)


if __name__ == "__main__":
    main()

#################################
