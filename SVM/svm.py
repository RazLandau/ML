#################################
# Your name: Raz Landau
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC, SVC

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C = 1000
    suppot_vectors = []
    svc_models = {
        'Lin': SVC(C=C, kernel="linear"),
        'Quad': SVC(C=C, kernel="poly", degree=2),
        'RBF': SVC(C=C),
    }
    for name, model in svc_models.items():
        model.fit(X_train, y_train)
        suppot_vectors.append(model.n_support_)
        create_plot(X_val, y_val, model)
        plt.savefig('q1a' + name + '.png')
        plt.clf()
    print(suppot_vectors)
    return suppot_vectors


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_vals = [np.math.pow(10, exp) for exp in np.arange(-5, 6)]
    train_accuracies, validation_accuracies = [], []
    for C in C_vals:
        svc = SVC(C=C, kernel="linear")
        svc.fit(X_train, y_train)
        train_accuracies.append(svc.score(X_train, y_train))
        validation_accuracies.append(svc.score(X_val, y_val))
        create_plot(X_val, y_val, svc)
        plt.savefig('q1b' + str(C) + '.png')
        plt.clf()

    plt.plot(C_vals, train_accuracies, 'r-')
    plt.plot(C_vals, validation_accuracies, 'b-')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.semilogx(10)
    axes = plt.gca()
    axes.set_ylim([
        np.min(train_accuracies + validation_accuracies) - 0.01,
        np.max(train_accuracies + validation_accuracies) + 0.01,
    ])
    plt.savefig('q1b.png')
    plt.clf()

    return validation_accuracies


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C = 10
    gamma_vals = [np.math.pow(10, exp) for exp in np.arange(-5, 6)]
    train_accuracies, validation_accuracies = [], []
    for gamma in gamma_vals:
        svc = SVC(C=C, gamma=gamma)
        svc.fit(X_train, y_train)
        train_accuracies.append(svc.score(X_train, y_train))
        validation_accuracies.append(svc.score(X_val, y_val))
        create_plot(X_val, y_val, svc)
        plt.savefig('q1c' + str(gamma) + '.png')
        plt.clf()

    plt.plot(gamma_vals, train_accuracies, 'r-')
    plt.plot(gamma_vals, validation_accuracies, 'b-')
    plt.grid()
    plt.xlabel('gamma')
    plt.ylabel('accuracy')
    plt.semilogx(10)
    axes = plt.gca()
    axes.set_ylim([
        np.min(train_accuracies + validation_accuracies) - 0.01,
        np.max(train_accuracies + validation_accuracies) + 0.01,
    ])
    plt.savefig('q1c.png')
    plt.clf()

    return validation_accuracies


def main():
    X_train, y_train, X_val, y_val = get_points()
    train_three_kernels(X_train, y_train, X_val, y_val)
    linear_accuracy_per_C(X_train, y_train, X_val, y_val)
    rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)


if __name__ == '__main__':
    main()