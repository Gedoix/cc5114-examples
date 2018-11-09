import math
import random

import numpy as np

import matplotlib.pyplot as plt

from src.basic_perceptrons.basic_perceptrons import Perceptron
from src.learning_perceptrons.basic_classifier import LinearClassifier


class SigmoidPerceptron(Perceptron):

    def __init__(self):
        """
        Default initializer for the perceptron, since it will only be used
        for testing
        """
        super().__init__()

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid function
        :param z:   Numeric array input
        :return:    Numeric array output (1/(1+e^-z))
        """
        return math.exp(-np.logaddexp(0, -z))

    def feed(self, inputs: np.ndarray) -> float:
        """
        Override of the feed method, using the sigmoid function
        :param inputs:  Inputs to the perceptron
        :return:        Output of the sigmoid function evaluating the weighted
        sum of inputs and bias
        """
        return self.sigmoid(self.evaluate(inputs))


class SigmoidClassifier(LinearClassifier):

    def __init__(self, slope: float, y_intercept: float):
        """
        Default initializer for the sigmoid classifier, since it will only
        be used for testing
        Classifier for classification based on a line
        Uses the Perceptron class from the basic_perceptron package
        :param slope:           Slope of the line
        :param y_intercept:     Interception of the line with the y axis
        """
        super().__init__(slope, y_intercept)
        self._perceptron = SigmoidPerceptron().built_with(weights_amount=2)


def classify_plot(train: int = 5000, point_generator_seed: int = 14509301):
    """
    Generates a plot marking in red all the wrongly classified points and in blue all of the successful ones
    :param train:                   Amount of training to be done
    :param point_generator_seed:    Seed for the pseudo-random number generator
    """
    slope = 2.031
    y_intercept = 12.576

    classifier = SigmoidClassifier(slope, y_intercept)

    np.random.seed(seed=point_generator_seed)

    classifier.auto_train(times=train)

    testing_points = 2000

    line_x = np.array(range(-200, 200))
    line_y = line_x*classifier.get_slope()+classifier.get_y_intercept()

    t_xs = []
    t_yr = []
    t_cs = []
    for i in range(testing_points):
        t_xs.append(random.uniform(-200, 200))
        t_yr.append(random.uniform(-200, 200))
        t_cs.append(classifier.expected_classification(t_xs[i], t_yr[i]))

    fig, ax = plt.subplots()
    ax.plot(line_x, line_y)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])

    accuracy = 0.0
    for i in range(testing_points):
        classification = classifier.classification(t_xs[i], t_yr[i])
        accuracy += 1.0 if classification == t_cs[i] else 0.0
        ax.scatter(t_xs[i], t_yr[i],
                   color='b' if classification == t_cs[i] else 'r',
                   label='correct' if classification == t_cs[i] else 'wrong')
    accuracy /= testing_points
    accuracy *= 100
    plt.title("Accuracy of "+str(int(accuracy))+"% for "+str(train)+" training examples")
    ax.grid(True)
    plt.show()


def accuracies_plot(train: int = 5000, automatic_points: bool = True, point_generator_seed: int = 14509301):
    """
    Generates a plot showing the accuracy of the classifier's output over
    amount of training examples
    :param train:                   Amount of training to be done
    :param automatic_points:        Whether or not the point should ignore the set seed
    :param point_generator_seed:    Seed for the pseudo-random number generator
    """
    slope = 2.031
    y_intercept = 12.576

    classifier = SigmoidClassifier(slope, y_intercept)

    testing_points = 1000
    np.random.seed(seed=point_generator_seed)
    randomize = np.random

    t_xs = []
    t_yr = []
    t_cs = []
    for i in range(testing_points):
        t_xs.append(randomize.uniform(-200, 200))
        t_yr.append(randomize.uniform(-200, 200))
        t_cs.append(classifier.expected_classification(t_xs[i], t_yr[i]))

    accuracies = []

    if automatic_points:

        for time in range(train + 1):

            accuracies.append(0)

            if time != 0:
                classifier.auto_train()

            for i in range(testing_points):
                t_cr = classifier.classification(t_xs[i], t_yr[i])
                accuracies[time] += 1 if t_cr == t_cs[i] else 0

            accuracies[time] *= 100 / testing_points

    else:

        xs = []
        yr = []
        cs = []
        for i in range(train):
            xs.append(randomize.uniform(-200, 200))
            yr.append(randomize.uniform(-200, 200))
            cs.append(classifier.expected_classification(xs[i], yr[i]))

        for time in range(train+1):

            accuracies.append(0)

            if time != 0:
                classifier.train(xs[time-1], yr[time-1], cs[time-1])

            for i in range(testing_points):
                t_cr = classifier.classification(t_xs[i], t_yr[i])
                accuracies[time] += 1 if t_cr == t_cs[i] else 0

            accuracies[time] *= 100 / testing_points

    fig, ax = plt.subplots()
    ax.plot(range(train+1), accuracies)
    ax.set_xlim([0, train])
    ax.set_ylim([0, 100])

    plt.title("Accuracy of classifier v/s times trained for a total of " + str(train) + " training examples")
    ax.grid(True)
    plt.show()


def main():
    classify_plot()
    accuracies_plot()


if __name__ == '__main__':
    main()
