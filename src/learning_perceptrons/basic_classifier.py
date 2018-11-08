import random

import numpy as np

import matplotlib.pyplot as plt

from basic_perceptrons.basic_perceptrons import Perceptron


class LinearClassifier:

    def __init__(self, slope: float, y_intercept: float):
        """
        Initializer for a simple classifier for classification based on a line
        Uses the Perceptron class from the basic_perceptron package
        :param slope:           Slope of the line
        :param y_intercept:     Interception of the line with the y axis
        """
        self.__slope = slope
        self.__y_intercept = y_intercept
        self._perceptron = Perceptron().built_with(weights_amount=2, learning_rate=0.05)

    def get_slope(self):
        """
        Returns slope of the expected classifier
        :return:    Analytical slope
        """
        return self.__slope

    def get_y_intercept(self):
        """
        Returns interception with the y axis for the expected classifier's line
        :return:    Analytical y interception
        """
        return self.__y_intercept

    def expected_classification(self, x: float, y: float):
        """
        Returns the expected classification of a point calculated from the analytical line given
        :param x:   X coordinate of the point
        :param y:   Y coordinate of the point
        :return:    Binary classification, 1 if below the line or 0 if above
        """
        return 1 if (y < (x*self.__slope+self.__y_intercept)) else 0

    def classification(self, x: float, y: float):
        """
        Returns the classification of a point calculated from using a learning perceptron
        :param x:   X coordinate of the point
        :param y:   Y coordinate of the point
        :return:    Binary classification, 1 if below the line or 0 if above
        """
        return 1 if self._perceptron.feed(np.array([x, y])) >= 0.5 else 0

    def train(self, x: float, y: float, expected: int, times: int = 1):
        """
        Automatically trains the classifier's perceptron based on a certain input and output for a specific
        amount of times
        :param x:           X coordinate of the point
        :param y:           Y coordinate of the point
        :param expected:    Expected classification for the point
        :param times:       Time to train on the example
        """
        while times != 0:
            self._perceptron.learn(expected, np.array([x, y]))
            times -= 1

    def auto_train(self, times: int = 1):
        """
        Automatically trains the classifier on randomized points using the analytical line
        :param times:   Time to train on an example (new example for every time)
        """
        while times != 0:
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            expected = self.expected_classification(x, y)
            self.train(x, y, expected)
            times -= 1


def classify_plot(train: int = 5000, point_generator_seed: int = 14509301):
    """
    Generates a plot marking in red all the wrongly classified points and in blue all of the successful ones
    :param train:                   Amount of training to be done
    :param point_generator_seed:    Seed for the pseudo-random number generator
    """
    slope = 2.031
    y_intercept = 12.576

    classifier = LinearClassifier(slope, y_intercept)

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

    classifier = LinearClassifier(slope, y_intercept)

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
