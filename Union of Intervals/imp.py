#################################
# Your name: Raz Landau
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
from scipy.stats import bernoulli
from intervalset import Interval, IntervalSet


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        return [
            (
                x,
                bernoulli.rvs(0.1) if (0.2 <= x <= 0.4) or (0.6 <= x <= 0.8) else bernoulli.rvs(0.8)
            )
            for x in list(np.random.sample(m))
        ]

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        sorted_samples = sorted(self.sample_from_D(m), key=lambda p: p[0])
        best_intervals = intervals.find_best_interval(
            [sample[0] for sample in sorted_samples],
            [sample[1] for sample in sorted_samples],
            k
        )[0]

        plt.figure()
        plt.plot([sample[0] for sample in sorted_samples], [sample[1] for sample in sorted_samples], '.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([0, 1, -0.1, 1.1])
        plt.xticks(np.arange(0, 1, 0.2))
        plt.gca().axvline(0.2)
        plt.gca().axvline(0.4)
        plt.gca().axvline(0.6)
        plt.gca().axvline(0.8)
        plt.gca().axhline(0)
        plt.gca().axhline(1)
        for interval in best_intervals:
            plt.hlines(0.5, interval[0], interval[1])
        plt.savefig('q1a.png')

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        errors = []
        for m in np.arange(m_first, m_last + 1, step):
            empirical_error = 0.0
            true_error = 0.0
            print("Experimenting for m =", m)
            for _ in np.arange(T):
                sorted_samples = sorted(self.sample_from_D(m), key=lambda p: p[0])
                best_intervals, best_error_count = intervals.find_best_interval(
                    [sample[0] for sample in sorted_samples],
                    [sample[1] for sample in sorted_samples],
                    k
                )
                empirical_error += best_error_count / m
                true_error += self.get_true_error(best_intervals)
            errors.append((empirical_error / T, true_error / T))

        plt.figure()
        plt.plot(np.arange(m_first, m_last + 1, step), [error[0] for error in errors], '.')
        plt.plot(np.arange(m_first, m_last + 1, step), [error[1] for error in errors], '.')
        plt.xlabel('m')
        plt.ylabel('error')
        plt.xticks(np.arange(m_first, m_last + 1, step))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid(True)
        plt.savefig('q1c.png')

        return errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        empirical_errors, true_errors = [], []
        sorted_samples = sorted(self.sample_from_D(m), key=lambda p: p[0])
        for k in np.arange(k_first, k_last + 1, step):
            print("Experimenting for k =", k)
            best_intervals, best_error_count = intervals.find_best_interval(
                [sample[0] for sample in sorted_samples],
                [sample[1] for sample in sorted_samples],
                k
            )
            empirical_errors.append(best_error_count / (m * 1.0))
            true_errors.append(self.get_true_error(best_intervals))

        plt.figure()
        plt.plot(np.arange(k_first, k_last + 1, step), empirical_errors, '.')
        plt.plot(np.arange(k_first, k_last + 1, step), true_errors, '.')
        plt.xlabel('k')
        plt.ylabel('error')
        plt.xticks(np.arange(k_first, k_last + 1, step))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid(True)
        plt.savefig('q1d.png')

        return empirical_errors.index(min(empirical_errors)) + 1

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        empirical_errors, true_errors, penalties = [], [], []
        sorted_samples = sorted(self.sample_from_D(m), key=lambda p: p[0])
        for k in np.arange(k_first, k_last + 1, step):
            print("Experimenting for k =", k)
            best_intervals, best_error_count = intervals.find_best_interval(
                [sample[0] for sample in sorted_samples],
                [sample[1] for sample in sorted_samples],
                k
            )
            empirical_errors.append(best_error_count / (m * 1.0))
            true_errors.append(self.get_true_error(best_intervals))
            penalties.append(np.sqrt((8 / m) * (2 * k * np.log((2 * np.e * m) / (2 * k)) + np.log(4 / 0.1))))
        sums = [empirical_error + penalty for empirical_error, penalty in zip(empirical_errors, penalties)]

        plt.figure()
        plt.plot(np.arange(k_first, k_last + 1, step), empirical_errors, '.')
        plt.plot(np.arange(k_first, k_last + 1, step), true_errors, '.')
        plt.plot(np.arange(k_first, k_last + 1, step), penalties, '.')
        plt.plot(np.arange(k_first, k_last + 1, step), sums, '.')
        plt.xlabel('k')
        plt.xticks(np.arange(k_first, k_last + 1, step))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid(True)
        plt.savefig('q1e.png')

        return sums.index(min(sums)) + 1

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        best_intervals, best_error = [], 1
        sorted_samples = sorted(self.sample_from_D(m), key=lambda p: p[0])
        for k in np.arange(1, 11, 1):
            print("Experimenting for k =", k)
            error = 0.0
            for t in np.arange(T):
                training_samples = [y for x, y in enumerate(sorted_samples) if x % 5 != t]
                holdout_samples = [y for x, y in enumerate(sorted_samples) if x % 5 == t]
                best_k_intervals = intervals.find_best_interval(
                    [sample[0] for sample in training_samples],
                    [sample[1] for sample in training_samples],
                    k
                )[0]
                error += (Assignment2.get_holdout_error(best_k_intervals, holdout_samples) / T)
            if error < best_error:
                best_intervals, best_error = best_k_intervals, error

        plt.figure()
        plt.plot([sample[0] for sample in sorted_samples], [sample[1] for sample in sorted_samples], '.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([0, 1, -0.1, 1.1])
        plt.xticks(np.arange(0, 1, 0.2))
        plt.gca().axvline(0.2)
        plt.gca().axvline(0.4)
        plt.gca().axvline(0.6)
        plt.gca().axvline(0.8)
        plt.gca().axhline(0)
        plt.gca().axhline(1)
        for interval in best_intervals:
            plt.hlines(0.5, interval[0], interval[1])
        plt.savefig('q1f.png')

        return len(best_intervals)

    #################################
    # Place for additional methods
    RANGE1 = IntervalSet(Interval(0, 0.2), Interval(0.4, 0.6), Interval(0.8, 1))
    RANGE2 = IntervalSet(Interval(0.2, 0.4), Interval(0.6, 0.8))

    def get_true_error(self, inters):
        error = 0.0
        inter_set = IntervalSet(*[
            Interval(inter[0], inter[1]) for inter in inters
        ])
        # false-positives
        for inter in inter_set & self.RANGE1:
            error += 0.2 * (inter[1] - inter[0])
        for inter in inter_set & self.RANGE2:
            error += 0.9 * (inter[1] - inter[0])
        # false-negatives
        complement_inter_set = IntervalSet(Interval(0, 1)) - inter_set
        for inter in complement_inter_set & self.RANGE1:
            error += 0.8 * (inter[1] - inter[0])
        for inter in complement_inter_set & self.RANGE2:
            error += 0.1 * (inter[1] - inter[0])
        return error

    @staticmethod
    def get_holdout_error(inters, holdouts):
        error = 0.0
        inter_set = IntervalSet(*[
            Interval(inter[0], inter[1]) for inter in inters
        ])
        for point in holdouts:
            point_inter = Interval(point[0], point[0])
            error += point[1] ^ (point_inter in inter_set)
        return error / len(holdouts)
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
