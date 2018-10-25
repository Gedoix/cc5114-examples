import math
import random

import numpy as np


class Perceptron:

    def __init__(self, input_amount: int, learning_rate: float = 0.1):
        self.__weights = 2 * np.random.random(input_amount) - 1.0
        self.__learning_rate = learning_rate
        self.__bias = random.uniform(-1.0, 1.0)
        self.__last_inputs = None
        self.__last_feed = None
        self.__delta = None

    def forward_propagate(self, inputs: np.ndarray):
        self.__last_inputs = inputs
        self.__last_feed = math.exp(-np.logaddexp(0,
                                                  -np.dot(inputs,
                                                          self.__weights) +
                                                  self.__bias))
        return self.__last_feed

    def back_propagate_from_error(self, error):
        self.__delta = error * np.dot(self.__last_feed, (1.0 - self.__last_feed))
        self.__weights += (self.__learning_rate *
                           self.__delta *
                           self.__last_inputs)
        self.__bias += self.__learning_rate * self.__delta

    def back_propagate(self, deltas, weights):
        error = np.dot(deltas, weights)
        self.__delta = error * np.dot(self.__last_feed, (1.0 - self.__last_feed))
        self.__weights += (self.__learning_rate *
                           self.__delta *
                           self.__last_inputs)
        self.__bias += self.__learning_rate * self.__delta

    def get_delta_and_weight(self, index: int):
        return self.__delta, self.__weights[index]


class Layer:

    def __init__(self, input_amount: int,
                 perceptron_amount: int,
                 learning_rate: float = 0.1):
        self.__perceptrons = []
        while perceptron_amount > 0:
            self.__perceptrons.append(Perceptron(input_amount,
                                                 learning_rate=learning_rate))
            perceptron_amount -= 1
        self.__last_inputs = None
        self.__last_feed = None
        self.__deltas = None

    def __getitem__(self, item):
        return self.__perceptrons[item]

    def __len__(self):
        return len(self.__perceptrons)

    def forward_propagate(self, inputs: np.ndarray):
        self.__last_inputs = inputs
        self.__last_feed = []
        for i in range(len(self.__perceptrons)):
            self.__last_feed.append(self.__perceptrons[i]
                                    .forward_propagate(inputs))
        self.__last_feed = np.array(self.__last_feed)
        return np.array(self.__last_feed)

    def back_propagate_from_error(self, error: np.ndarray):
        for i in range(len(self.__perceptrons)):
            self.__perceptrons[i].back_propagate_from_error(error)

    def back_propagate(self, last_layer):

        for i in range(len(self.__perceptrons)):
            deltas, weights = last_layer.get_deltas_and_weights(i)
            self.__perceptrons[i].back_propagate(deltas, weights)

    def get_deltas_and_weights(self):
        deltas = []
        weights = []
        for i in range(len(self.__perceptrons)):
            delta, weight = self[i].get_delta_and_weight()
            deltas.append(delta)
            weights.append(weight)
        return deltas, weights


class Network:

    def __init__(self, input_amount: int,
                 perceptron_per_hidden_layer_amounts: list,
                 output_amount: int,
                 learning_rate: float = 0.1):
        self.__hidden_layers = [
            Layer(input_amount,
                  perceptron_per_hidden_layer_amounts[0],
                  learning_rate=learning_rate)
        ]
        for i in range(len(perceptron_per_hidden_layer_amounts) - 1):
            self.__hidden_layers.append(
                Layer(perceptron_per_hidden_layer_amounts[i],
                      perceptron_per_hidden_layer_amounts[i + 1],
                      learning_rate=learning_rate))
        self.__output_layer = Layer(perceptron_per_hidden_layer_amounts[-1],
                                    output_amount,
                                    learning_rate=learning_rate
                                    )
        self.__results = None
        self.__mean_true_positives = 0
        self.__mean_false_negatives = 0
        self.__mean_true_negatives = 0
        self.__mean_false_positives = 0
        self.__mean_absolute_error = 0
        self.__mean_squared_error = 0

    def forward_propagate(self, inputs: np.ndarray):
        self.__results = self.__hidden_layers[0].forward_propagate(inputs)
        for i in range(len(self.__hidden_layers) - 1):
            self.__results = self.__hidden_layers[i + 1] \
                .forward_propagate(self.__results)
        self.__results = 1.0 if self.__output_layer.forward_propagate(self.__results) >= 0.5 else 0.0
        return self.__results

    def back_propagare(self, expected_outputs: np.ndarray):
        error = expected_outputs - self.__results
        self.__output_layer.back_propagate_from_error(error)
        last_layer = self.__output_layer
        for i in range(len(self.__hidden_layers)):
            j = len(self.__hidden_layers) - i - 1
            self.__hidden_layers[j].back_propagate(last_layer)
            last_layer = self.__hidden_layers[j]

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        self.forward_propagate(inputs)
        self.back_propagare(outputs)

    def generate_metrics(self, inputs: np.ndarray, outputs: np.ndarray):
        self.__mean_true_positives = 0
        self.__mean_true_negatives = 0
        self.__mean_false_negatives = 0
        self.__mean_false_positives = 0
        self.__mean_absolute_error = 0
        self.__mean_squared_error = 0
        self.forward_propagate(inputs)
        for i in range(len(outputs)):
            self.__mean_absolute_error += (outputs[i] - self.__results[i])
            self.__mean_absolute_error += (outputs[i] - self.__results[i]) ^ 2
            if self.__results[i] == outputs[i]:
                if outputs[i] == 1:
                    self.__mean_true_positives += 1
                else:
                    self.__mean_true_negatives += 1
            else:
                if outputs[i] == 1:
                    self.__mean_false_negatives += 1
                else:
                    self.__mean_false_positives += 1
        self.__mean_true_positives /= len(outputs)
        self.__mean_true_negatives /= len(outputs)
        self.__mean_false_negatives /= len(outputs)
        self.__mean_false_positives /= len(outputs)
        self.__mean_absolute_error /= len(outputs)
        self.__mean_absolute_error /= len(outputs)

    def get_accuracy(self):
        return (self.__mean_true_positives +
                self.__mean_true_negatives) / (
                       self.__mean_true_positives +
                       self.__mean_true_negatives +
                       self.__mean_false_positives +
                       self.__mean_false_negatives
               )

    def get_precision(self):
        return self.__mean_true_positives / (
                self.__mean_true_positives +
                self.__mean_false_positives
        )

    def get_recall(self):
        return self.__mean_true_positives / (
                self.__mean_true_positives +
                self.__mean_false_negatives
        )

    def get_specificity(self):
        return self.__mean_true_negatives / (
                self.__mean_true_negatives +
                self.__mean_false_positives
        )

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2 * r * p / (r + p)


def main(training_amount: int=10000, testing_amount: int=500):
    network = Network(2, [2, 3, 2], 1)
    training_points = np.random.uniform(-100, 100, (2, training_amount))
    testing_points = np.random.uniform(-100, 100, (2, testing_amount))
    slope = 2.0
    y_intercept = -1.0
    training_classes = np.greater_equal(training_points[1, :], y_intercept+slope*training_points[0, :])
    testing_classes = np.greater_equal(testing_points[1, :], y_intercept+slope*testing_points[0, :])

    for i in range(training_amount):
        network.train(training_points[:, i], training_classes[i])
        p = 0
        for i in range(testing_amount):
            network.generate_metrics(testing_points[:, i], testing_classes[i])
            p += network.get_precision()
        p /= testing_amount
        print("Precision for step ", i, " equals = ", p)


if __name__ == '__main__':
    main()
