import math
import random

import numpy as np

import matplotlib.pyplot as plt

from session_1.basic_perceptrons import Perceptron
from session_2.basic_classifier import LinearClassifier


class SigmoidPerceptron(Perceptron):

    def __init__(self):
        super().__init__()

    @staticmethod
    def sigmoid(z):
        return math.exp(-np.logaddexp(0, -z))

    def feed(self, inputs: np.ndarray):
        return self.sigmoid(self.evaluate(inputs))


class SigmoidClassifier(LinearClassifier):

    def __init__(self, slope: float, y_intercept: float):
        super().__init__(slope, y_intercept)
        self._perceptron = SigmoidPerceptron().built_with(weights_amount=2)


def perceptron_logical_accuracies(train: int = 1000):
    classifier = SigmoidClassifier(2.031, 12.576)
    classifier.auto_train(times=train)
    xs = np.array(range(-100, 100))
    ys = xs * classifier.get_slope() + classifier.get_y_intercept()
    yr = []
    for i in range(200):
        yr.append(random.uniform(-200, 200))
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    accuracy = 0
    for i in range(200):
        classification = classifier.classification(xs[i], yr[i])
        expected_classification = classifier.expected_classification(xs[i], yr[i])
        accuracy += 1 if classification == expected_classification else 0
        ax.scatter(xs[i], yr[i],
                   color='b' if classification == expected_classification else 'r',
                   label='correct' if classification == expected_classification else 'wrong')
    accuracy /= 2
    plt.title("accuracy of " + str(accuracy) + "% for " + str(train) + " training examples")
    ax.grid(True)
    plt.show()


def sigmoid_logical_accuracies(train: int = 1000, training_points: int = 1000, randomized: bool = True,
                               testing_points: int = 100):
    classifier = SigmoidClassifier(2.031, 12.576)

    xs = np.array(range(int(-training_points / 2), int(training_points / 2) + 1))
    yr = []
    cs = []
    for i in range(training_points):
        yr.append(random.uniform(-200, 200))
        cs.append(classifier.expected_classification(xs[i], yr[i]))

    t_xs = np.array(range(int(-testing_points / 2), int(testing_points / 2) + 1))
    t_yr = []
    t_cs = []
    for i in range(testing_points):
        t_yr.append(random.uniform(-200, 200))
        t_cs.append(classifier.expected_classification(t_xs[i], t_yr[i]))

    accuracies = []
    for time in range(train+1):

        accuracies.append(0)

        if time != 0:
            for i in range(training_points):
                classifier.train(xs[i], yr[i], cs[i])

        for i in range(testing_points):
            t_cr = classifier.classification(t_xs[i], t_yr[i])
            accuracies[time] += 1 if t_cr == t_cs[i] else 0

        accuracies[time] *= 100 / testing_points

        if randomized:
            for i in range(training_points):
                yr[i] = random.uniform(-200, 200)
                cs[i] = classifier.expected_classification(xs[i], yr[i])

    fig, ax = plt.subplots()
    ax.plot(range(train+1), accuracies)

    plt.title("Accuracy of classifier v/s times trained for a total of " + str(train*training_points if randomized else
                                                                               train) + " times")
    ax.grid(True)
    plt.show()


def main():
    perceptron_logical_accuracies()
    sigmoid_logical_accuracies()


if __name__ == '__main__':
    main()
