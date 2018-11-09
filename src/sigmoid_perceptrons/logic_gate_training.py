import numpy as np

import matplotlib.pyplot as plt

from src.learning_perceptrons.basic_classifier import LinearClassifier
from src.sigmoid_perceptrons.sigmoid_perceptrons import SigmoidClassifier


def accuracies_plot(train: int = 5000):
    """
    Generates a plot showing the accuracy of the classifier's output over
    amount of training examples
    :param train:                   Amount of training to be done
    """
    inputs = [np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
    outputs = {"and": [0, 0, 0, 1], "or": [0, 1, 1, 1], "nand": [1, 1, 1, 0],
               "xor": [0, 1, 1, 0]}
    learners = {
        "and": [LinearClassifier(0, 0), SigmoidClassifier(0, 0)],
        "or": [LinearClassifier(0, 0), SigmoidClassifier(0, 0)],
        "nand": [LinearClassifier(0, 0), SigmoidClassifier(0, 0)],
        "xor": [LinearClassifier(0, 0), SigmoidClassifier(0, 0)]}

    accuracies = {"and": [], "or": [], "nand": [], "xor": []}
    sigmoid_accuracies = {"and": [], "or": [], "nand": [], "xor": []}

    for time in range(train + 1):

        if time != 0:
            for key in outputs.keys():
                for i in range(4):
                    learners.get(key)[0].train(inputs[i][0], inputs[i][1], outputs.get(key)[i])
                    learners.get(key)[1].train(inputs[i][0], inputs[i][1], outputs.get(key)[i])

        for key in outputs.keys():
            accuracies.get(key).append(0)
            sigmoid_accuracies.get(key).append(0)
            for i in range(4):
                result = learners.get(key)[0].classification(inputs[i][0], inputs[i][1])
                sigmoid_result = learners.get(key)[1].classification(inputs[i][0], inputs[i][1])
                accuracies.get(key)[time] += 1 if result == outputs.get(key)[i] else 0
                sigmoid_accuracies.get(key)[time] += 1 if sigmoid_result == outputs.get(key)[i] else 0

            accuracies.get(key)[time] *= 100 / 4
            sigmoid_accuracies.get(key)[time] *= 100 / 4

    fig, ax = plt.subplots()
    for key in outputs.keys():
        ax.plot(range(train + 1), accuracies.get(key), label="Normal "+key)
        ax.plot(range(train + 1), sigmoid_accuracies.get(key), label="Sigmoid "+key)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlim([0, train])
    ax.set_ylim([0, 100])

    plt.title("Accuracy of classifier v/s times trained for a total of " + str(train) + " training examples")
    ax.grid(True)
    plt.show()


def main():
    accuracies_plot()


if __name__ == '__main__':
    main()
