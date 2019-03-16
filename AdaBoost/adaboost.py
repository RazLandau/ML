#################################
# Your name: Raz Landau
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)
    D = (1.0 / n) * np.ones(n)
    hypotheses = []
    alpha_vals = []
    for t in range(T):
        print("t =", t)
        hypothesis, error = get_hypothesis_and_error(X_train, y_train, D)
        hypotheses.append(hypothesis)
        alpha = 0.5 * np.log((1.0 - error) / error)
        alpha_vals.append(alpha)
        D = get_new_distribution(D, X_train, y_train, alpha, hypothesis)
    return hypotheses, alpha_vals

##############################################
# You can add more methods here, if needed.


def get_hypothesis_and_error(X_train, y_train, D):
    down_stump_erm_j, down_stump_erm_theta, down_stump_erm_error = erm_for_decision_stumps(X_train, y_train, D, b=1)
    up_stump_erm_j, up_stump_erm_theta, up_stump_erm_error = erm_for_decision_stumps(X_train, y_train, D, b=-1)
    if down_stump_erm_error < up_stump_erm_error:
        return (1, down_stump_erm_j, down_stump_erm_theta), down_stump_erm_error
    else:
        return (-1, up_stump_erm_j, up_stump_erm_theta), up_stump_erm_error


def erm_for_decision_stumps(X_train, y_train, D, b):
    f_star = float('inf')
    theta_star = float('inf')
    j_star = 0
    m = X_train.shape[0]
    d = X_train[0].shape[0]
    for j in range(d):
        s = sorted(list(zip(X_train, y_train, D)), key=lambda x: x[0][j])
        s.append(((s[-1][0][j] + 1) * np.ones(d), None, None))
        f = np.sum(D[np.argwhere(y_train == b)])
        if f < f_star:
            f_star = f
            theta_star = s[0][0][j] - 1
            j_star = j
        for i in range(m):
            f = f - b * np.dot(s[i][1], s[i][2])
            if f < f_star and s[i][0][j] != s[i+1][0][j]:
                f_star = f
                theta_star = 0.5 * (s[i][0][j] + s[i+1][0][j])
                j_star = j
    return j_star, theta_star, f_star


def predict(hypothesis, x):
    b, j, theta = *hypothesis,
    return np.sign(theta - x[j]) * b


def get_new_distribution(D, X_train, y_train, alpha, hypothesis):
    softmax_vals = np.array([np.exp(-y_train[i] * alpha * predict(hypothesis, X_train[i])) for i in range(len(D))])
    return softmax_vals * D / np.dot(D, softmax_vals)


def get_error(samples, labels, classifier):
    error = 0.0
    for sample, label in zip(samples, labels):
        error += classifier(sample) != label
    return error / len(samples)


def get_classifier(alpha_vals, hypotheses, t):
    return lambda x: np.sign(np.sum([predict(hypotheses[i], x) * alpha_vals[i] for i in range(t)]))


def get_loss(samples, labels, alpha_vals, hypotheses, t):
    return np.average([
        np.exp(
            -label * get_classifier(alpha_vals, hypotheses, t)(sample)
        ) for sample, label in zip(samples, labels)
    ])


def plot_two_lines(x_vals, y1_vals, y2_vals, x_label, y_label, file_name, line1_label='', line2_label='',
                   x_ticks=None, y_ticks=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if x_ticks:
        plt.xticks(np.arange(min(x_vals) - 1, max(x_vals) + 1, x_ticks))
    if y_ticks:
        plt.yticks(np.arange(min(y1_vals + y2_vals) - 1, max(y1_vals + y2_vals) + 1, y_ticks))
    ax.set_ylim([min(y1_vals + y2_vals) - 0.01, max(y1_vals + y2_vals) + 0.01])
    ax.plot(x_vals, y1_vals, 'r-', label=line1_label)
    ax.plot(x_vals, y2_vals, 'b-', label=line2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if line1_label and line2_label:
        ax.legend()
    fig.savefig(file_name)
    fig.clf()
##############################################


def q1a(T, alpha_vals, hypotheses, X_train, y_train, X_test, y_test):
    training_errors = []
    test_errors = []
    for t in range(T):
        classifier = get_classifier(alpha_vals, hypotheses, t)
        training_errors.append(get_error(X_train, y_train, classifier))
        test_errors.append(get_error(X_test, y_test, classifier))
    plot_two_lines(np.arange(T), training_errors, test_errors, 'T', 'Errors', 'q1a.png',
                   line1_label='Training Errors', line2_label='Test Errors')


def q1b(T, hypotheses, vocab):
    for t in range(T):
        b, j, theta = *hypotheses[t],
        print("WL{0}: count of {1} is at {2} {3}".format(t + 1, vocab[j], "most" if b > 0 else "least", theta))


def q1c(T, alpha_vals, hypotheses, X_train, y_train, X_test, y_test):
    training_losses = []
    test_losses = []
    for t in range(T):
        training_losses.append(get_loss(X_train, y_train, alpha_vals, hypotheses, t))
        test_losses.append(get_loss(X_test, y_test, alpha_vals, hypotheses, t))
    plot_two_lines(np.arange(T), training_losses, test_losses, 'T', 'Losses', 'q1c.png',
                   line1_label='Training Losses', line2_label='Test Losses')


def main():
    T = 80
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    ##############################################
    # You can add more methods here, if needed.
    q1a(T, alpha_vals, hypotheses, X_train, y_train, X_test, y_test)
    q1b(10, hypotheses, vocab)
    q1c(T, alpha_vals, hypotheses, X_train, y_train, X_test, y_test)
    ##############################################


if __name__ == '__main__':
    main()



