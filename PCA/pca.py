import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plot_vector_as_image(image, h, w, title):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    # plt.show()
    plt.savefig(title)


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.reshape(h * w)
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################


"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
    U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
    of the covariance matrix.
    S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    _, s, vh = np.linalg.svd(X)
    U = vh[:k]
    S = s[:k]**2
    return U, S


def q1b():
    selected_images, h, w = get_pictures_by_name()
    U = PCA(selected_images, 10)[0]
    for i in range(10):
        plot_vector_as_image(U[i], h, w, F'q1b{i+1}.png')


def q1c():
    k_values = [1, 5, 10, 30, 50, 100]
    selected_images, h, w = get_pictures_by_name()
    U, S = PCA(selected_images, k_values[-1])
    for i in range(1, 6):
        print(F'i={i}')
        selected_image = selected_images[i]
        plt.imshow(selected_image.reshape((h, w)), cmap=plt.cm.gray)
        plt.savefig(F'q1c{i}original')
        distances = []
        for k in k_values:
            k_dim_selected_image = U[:k].T @ U[:k] @ selected_image
            plt.imshow(k_dim_selected_image.reshape((h, w)), cmap=plt.cm.gray)
            plt.savefig(F'q1c{i}transformed{k}')
            distances.append(np.linalg.norm(selected_image-k_dim_selected_image))
        plt.figure()
        plt.plot(k_values, distances)
        plt.xlabel('k')
        plt.ylabel('L2 distance')
        plt.savefig(F'q1c{i}distances.png')
        plt.clf()


def q1d():
    accuracies = []
    k_values = [1, 5, 10, 30, 50, 100, 150, 300]
    X, y = fetch_lfw_people(min_faces_per_person=70, resize=0.4, return_X_y=True)
    U, S = PCA(X, k_values[-1])
    for k in k_values:
        print(F'k={k}')
        X_train, X_test, y_train, y_test = train_test_split(
            [U[:k] @ x for x in X], y, test_size=0.25)
        svc = SVC(C=1000, gamma=1e-7)
        svc.fit(X_train, y_train)
        accuracies.append(svc.score(X_test, y_test))
    plt.figure()
    plt.plot(k_values, accuracies)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.savefig('q1d.png')
    plt.clf()


def main():
    q1b()
    q1c()
    q1d()


if __name__ == "__main__":
    main()
