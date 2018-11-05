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
        # Inputs are saved
        self.__last_inputs = inputs

        # Sigmoid function is calculated
        self.__last_feed = math.exp(-np.logaddexp(0.0,
                                                  -np.dot(inputs,
                                                          self.__weights) +
                                                  self.__bias))
        assert isinstance(self.__last_feed, float)
        return self.__last_feed

    def back_propagate_from_error(self, error):
        self.__delta = (error * np.dot(self.__last_feed, (1.0 - self.__last_feed)))[0]
        self.__weights += (self.__learning_rate *
                           self.__delta *
                           self.__last_inputs)
        self.__bias += self.__learning_rate * self.__delta

    def back_propagate(self, deltas, weights):
        error = float(np.dot(deltas, weights))
        assert isinstance(error, float)
        self.__delta = (error * (self.__last_feed * (1.0 - self.__last_feed)))
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
        for perceptron in self.__perceptrons:
            self.__last_feed.append(perceptron.forward_propagate(inputs))
        self.__last_feed = np.array(self.__last_feed)
        return np.array(self.__last_feed)

    def back_propagate_from_error(self, error: np.ndarray):
        for perceptron in self.__perceptrons:
            perceptron.back_propagate_from_error(error)

    def back_propagate(self, last_layer):
        for index, perceptron in enumerate(self.__perceptrons):
            deltas, weights = last_layer.get_deltas_and_weights(index)
            perceptron.back_propagate(deltas, weights)

    def get_deltas_and_weights(self, input_index):
        deltas = []
        weights = []
        for perceptron in self.__perceptrons:
            delta, weight = perceptron.get_delta_and_weight(input_index)
            deltas.append(delta)
            weights.append(weight)
        return deltas, weights


class Network:

    def __init__(self, input_amount: int,
                 perceptron_per_hidden_layer_amounts: list,
                 output_amount: int,
                 learning_rate: float = 0.1):

        # A first hidden layer is added
        self.__hidden_layers = [
            Layer(input_amount,
                  perceptron_per_hidden_layer_amounts[0],
                  learning_rate=learning_rate)
        ]

        # The rest of the hidden layers are added
        for i in range(len(perceptron_per_hidden_layer_amounts) - 1):
            self.__hidden_layers.append(
                Layer(perceptron_per_hidden_layer_amounts[i],
                      perceptron_per_hidden_layer_amounts[i + 1],
                      learning_rate=learning_rate))

        # The output layer is added
        self.__output_layer = Layer(perceptron_per_hidden_layer_amounts[-1],
                                    output_amount,
                                    learning_rate=learning_rate)

        # Auxiliary variables for learning are initialized to their defaults
        self.__results = None
        self.__mean_true_positives = 0.0
        self.__mean_false_negatives = 0.0
        self.__mean_true_negatives = 0.0
        self.__mean_false_positives = 0.0
        self.__mean_absolute_error = 0.0
        self.__mean_squared_error = 0.0

    def forward_propagate(self, inputs: np.ndarray):

        # The results propagate through the first layer
        self.__results = self.__hidden_layers[0].forward_propagate(inputs)

        # The results propagate through the rest of the layers
        for i in range(len(self.__hidden_layers) - 1):
            self.__results = self.__hidden_layers[i + 1] \
                .forward_propagate(self.__results)

        # The results of the last layer are extracted
        self.__results = self.__output_layer.forward_propagate(self.__results).round(0)
        return self.__results

    def back_propagate(self, expected_outputs: np.ndarray):

        # The error of the output layer is found
        error = expected_outputs - self.__results

        # The output layer learns through back propagation
        self.__output_layer.back_propagate_from_error(error)

        # The last back propagated layer is saved
        last_layer = self.__output_layer

        # Back propagation goes over all hidden layers
        for i in range(len(self.__hidden_layers)):
            # Layers are checked from last to first
            j = len(self.__hidden_layers) - i - 1

            # Back propagation happens, giving access to each layer to the last
            # layer that has already learned
            self.__hidden_layers[j].back_propagate(last_layer)

            # The last back propagated layer is saved
            last_layer = self.__hidden_layers[j]

    def train(self, inputs: np.ndarray, outputs: np.ndarray):

        # Inputs are analyzed
        self.forward_propagate(inputs)

        # The network learns
        self.back_propagate(outputs)

    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray):
        assert np.size(inputs[0, :]) == np.size(outputs[0, :])
        for i in range(np.size(inputs[0, :])):
            self.train(inputs[:, i], outputs[:, i])

    def generate_metrics_epoch(self, inputs: np.ndarray, outputs: np.ndarray):
        assert np.size(inputs[0, :]) == np.size(outputs[0, :])
        self.__mean_true_positives = 0.0
        self.__mean_true_negatives = 0.0
        self.__mean_false_negatives = 0.0
        self.__mean_false_positives = 0.0
        self.__mean_absolute_error = 0.0
        self.__mean_squared_error = 0.0
        for i in range(np.size(inputs[0, :])):
            self.forward_propagate(inputs[:, i])
            for index, output in enumerate(outputs[:, i]):
                self.__mean_absolute_error += np.sum(np.subtract(output, self.__results[index]))
                self.__mean_squared_error += np.sum(np.subtract(output, self.__results[index])) ** 2.0
                if self.__results[index] == output:
                    if output == 1.0:
                        self.__mean_true_positives += 1.0
                    else:
                        self.__mean_true_negatives += 1.0
                else:
                    if output == 1.0:
                        self.__mean_false_negatives += 1.0
                    else:
                        self.__mean_false_positives += 1.0
        self.__mean_true_positives /= np.size(inputs[0, :])
        self.__mean_true_negatives /= np.size(inputs[0, :])
        self.__mean_false_negatives /= np.size(inputs[0, :])
        self.__mean_false_positives /= np.size(inputs[0, :])
        self.__mean_absolute_error /= np.size(inputs[0, :])
        self.__mean_squared_error /= np.size(inputs[0, :])

    def get_accuracy(self):
        return 0.0 if (self.__mean_true_positives + self.__mean_true_negatives) == 0.0 else (
                (self.__mean_true_positives + self.__mean_true_negatives) / (
                 self.__mean_true_positives +
                 self.__mean_true_negatives +
                 self.__mean_false_positives +
                 self.__mean_false_negatives))

    def get_precision(self):
        return 0.0 if self.__mean_true_positives == 0.0 else (
                self.__mean_true_positives / (self.__mean_true_positives +
                                              self.__mean_false_positives))

    def get_recall(self):
        return 0.0 if self.__mean_true_positives == 0.0 else (
                self.__mean_true_positives / (self.__mean_true_positives +
                                              self.__mean_false_negatives))

    def get_specificity(self):
        return 0.0 if self.__mean_true_negatives == 0.0 else (
                self.__mean_true_negatives / (self.__mean_true_negatives +
                                              self.__mean_false_positives))

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2 * r * p / (r + p)


def main(training_points_amount: int = 1000, testing_points_amount: int = 200, training_epochs: int = 100):
    network = Network(2, [1], 1, 0.3)

    training_points = np.random.uniform(-100, 100, (2, training_points_amount))
    testing_points = np.random.uniform(-100, 100, (2, testing_points_amount))

    slope = 2.0
    y_intercept = -1.0

    one_array = np.array([1.0])
    zero_array = np.array([0.0])

    training_classes = np.zeros((1, training_points_amount))
    for i, c in enumerate(training_classes[0, :]):
        training_classes[0, i] = one_array if training_points[1, i] > y_intercept + slope * training_points[0, i] \
            else zero_array

    testing_classes = np.zeros((1, testing_points_amount))
    for i, c in enumerate(testing_classes[0, :]):
        testing_classes[0, i] = one_array if training_points[1, i] > y_intercept + slope * training_points[0, i] \
            else zero_array

    for i in range(training_epochs):
        network.train_epoch(training_points, training_classes)
        network.generate_metrics_epoch(testing_points, testing_classes)
        a = network.get_accuracy()
        p = network.get_precision()
        r = network.get_recall()
        s = network.get_specificity()
        print("--------------------------------------------------------------")
        print("Accuracy for epoch ", i + 1, " equals = ", a)
        print("Precision for epoch ", i + 1, " equals = ", p)
        print("Recall for epoch ", i + 1, " equals = ", r)
        print("Specificity for epoch ", i + 1, " equals = ", s)
        print("--------------------------------------------------------------")


if __name__ == '__main__':
    main()
