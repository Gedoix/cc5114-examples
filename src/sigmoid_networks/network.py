import math
import random

import numpy as np

import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, input_amount: int, learning_rate: float = 0.1):
        """
        Constructor for a single perceptron in the network
        :param input_amount:    Length of the input vector received by the
        network
        :param learning_rate:   Rate of learning shared by all perceptron in
         the network
        """
        self.__weights = 2 * np.random.random(input_amount) - 1.0
        self.__learning_rate = learning_rate
        self.__bias = random.uniform(-1.0, 1.0)
        self.__last_inputs = None
        self.__last_feed = None
        self.__delta = None

    def forward_propagate(self, inputs: np.ndarray) -> float:
        """
        Produces an output from the perceptron's values
        :rtype: float
        :param inputs:  Vector of inputs for the perceptron to process
        :return:        Single float value output
        """
        # Inputs are saved
        self.__last_inputs = inputs

        # Sigmoid function is calculated
        self.__last_feed = math.exp(-np.logaddexp(0.0,
                                                  -np.dot(inputs,
                                                          self.__weights) +
                                                  self.__bias))
        return self.__last_feed

    def back_propagate_output_layer(self, error: float) -> None:
        """
        Single step of the error back-propagation algorithm
        Special case for output-layer perceptrons
        :rtype: None
        :param error:   Error of the perceptron's original output
        """
        self.__delta = error * self.__last_feed * (1.0 - self.__last_feed)

    def back_propagate_from_weight(self, deltas: np.ndarray, weights: np.ndarray) -> None:
        """
        Single step of the error back-propagation algorithm
        General case for hidden-layer perceptrons
        :param deltas:  Delta values calculated by the next layer
        :param weights: Weight values used by the next layer to evaluate this
        perceptron's output
        """
        error = float(np.dot(deltas, weights))
        assert isinstance(error, float)
        self.__delta = error * (self.__last_feed * (1.0 - self.__last_feed))

    def update_values(self) -> None:
        """
        Updates weight and bias values for the network, intended for use after
        a single back propagation
        """
        self.__weights += (self.__learning_rate *
                           self.__delta *
                           self.__last_inputs)
        self.__bias += self.__learning_rate * self.__delta

    def get_delta_and_weight(self, index: int) -> (float, float):
        """
        Returns the perceptron's delta value, along with a single weight, the
        one specified by the given index
        :param index:   Index of the weight to return
        :return:        Delta and weight values
        """
        return self.__delta, self.__weights[index]


class Layer:

    def __init__(self, input_amount: int,
                 perceptron_amount: int,
                 learning_rate: float = 0.1):
        """
        Constructor for a layer of perceptrons
        :param input_amount:        Amount of inputs the perceptrons of this
        layer will individually accept
        :param perceptron_amount:   Amount of parallel perceptrons in this layer
        :param learning_rate:       Learning rate for all perceptrons within
        """
        self.__perceptrons = []
        while perceptron_amount > 0:
            self.__perceptrons.append(Perceptron(input_amount,
                                                 learning_rate=learning_rate))
            perceptron_amount -= 1
        self.__last_inputs = None
        self.__last_feed = None
        self.__deltas = None

    def __getitem__(self, index: int) -> Perceptron:
        """
        Gets the perceptron specified by the index
        :param index:   Index of the needed perceptron
        :return:        Perceptron at said index
        """
        return self.__perceptrons[index]

    @property
    def __len__(self) -> int:
        """
        Gets amount of perceptrons in the layer
        :return:    Amount of perceptrons in the layer
        """
        return len(self.__perceptrons)

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Produces the outputs of all perceptrons within the layer
        :param inputs:  Input vector for the layer
        :return:        Output vector of the perceptron outputs
        """
        self.__last_inputs = inputs
        self.__last_feed = []
        for perceptron in self.__perceptrons:
            self.__last_feed.append(perceptron.forward_propagate(inputs))
        self.__last_feed = np.array(self.__last_feed)
        return np.array(self.__last_feed)

    def back_propagate_output_layer(self, error: np.ndarray) -> None:
        """
        Single layer-step of the error back-propagation algorithm
        Special case for the output-layer
        :param error: Vector of errors of the layer's original outputs
        """
        for i, perceptron in enumerate(self.__perceptrons):
            perceptron.back_propagate_output_layer(error[i])

    def back_propagate(self, next_layer) -> None:
        """
        Single layer-step of the error back-propagation algorithm
        General case for hidden layers
        :param next_layer: Next layer in the network, must have just been
        back-propagated
        """
        for index, perceptron in enumerate(self.__perceptrons):
            deltas, weights = next_layer.get_deltas_and_weights(index)
            perceptron.back_propagate_from_weight(deltas, weights)

    def update_values(self) -> None:
        """
        Updates weight and bias values for all perceptrons in the layer
        Intended for use after a back-propagation
        """
        for perceptron in self.__perceptrons:
            perceptron.update_values()

    def get_deltas_and_weights(self, input_index: int) -> (np.ndarray, np.ndarray):
        """
        Returns all delta values of the perceptrons in the layer along their
        weight values for the perceptron in the previous layer's specified
        index
        :param input_index:     Index in the previous layer for getting the
        weights
        :return:                Deltas and weights
        """
        deltas = []
        weights = []
        for perceptron in self.__perceptrons:
            delta, weight = perceptron.get_delta_and_weight(input_index)
            deltas.append(delta)
            weights.append(weight)
        return np.array(deltas), np.array(weights)


class Network:

    def __init__(self, input_amount: int,
                 perceptron_per_hidden_layer_amounts: list,
                 output_amount: int,
                 learning_rate: float = 0.1):
        """
        Constructor for a perceptron network
        :param input_amount:                        Amount of inputs the
        network will accept
        :param perceptron_per_hidden_layer_amounts: List of amounts of
        perceptrons per layer
        :param output_amount:                       Length of the output vector
        of the network
        :param learning_rate:                       Learning rate for all
        perceptrons in the network
        """
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

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Produces the output of the network by evaluating all layers within
        :param inputs:  Input vector of the network
        :return:        Output vector of the network
        """
        # The results propagate through the first layer
        self.__results = self.__hidden_layers[0].forward_propagate(inputs)

        # The results propagate through the rest of the layers
        for i in range(len(self.__hidden_layers) - 1):
            self.__results = self.__hidden_layers[i + 1] \
                .forward_propagate(self.__results)

        # The results of the last layer are extracted
        self.__results = self.__output_layer.forward_propagate(self.__results)

        # The results are turned to binary
        for i, result in enumerate(self.__results):
            self.__results[i] = 1.0 if result >= 0.5 else 0.0

        return self.__results

    def back_propagate(self, expected_outputs: np.ndarray) -> None:
        """
        Propagates the network's error derivative through all layers
        :param expected_outputs:    Output expected for the given input
        """
        # The error of the output layer is found
        error = expected_outputs - self.__results

        # The output layer learns through back propagation
        self.__output_layer.back_propagate_output_layer(error)

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

    def update_values(self) -> None:
        """
        Updates internal values within the network
        """
        for layer in self.__hidden_layers:
            layer.update_values()
        self.__output_layer.update_values()

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Automatically trains the network for an input-output pair
        :param inputs:  Input vector fot the network
        :param outputs: Expected output vector for the network
        """
        # Inputs are analyzed
        self.forward_propagate(inputs)

        # The network evaluates it's life mistakes
        self.back_propagate(outputs)

        # The network learns
        self.update_values()

    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """
        Automatically trains the network for an array of inputs and an equally
        sized array of expected outputs
        :param inputs:  Array of vector inputs for the network
        :param outputs: Array of vector expected outputs for the network
        """
        assert np.size(inputs[0, :]) == np.size(outputs[0, :])
        for i in range(np.size(inputs[0, :])):
            self.train(inputs[:, i], outputs[:, i])

    def generate_metrics_epoch(self, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Automatically tests the network's efficiency at prediction with an
        array of vector inputs and an array of vector outputs
        :param inputs:  Array of vector inputs for the network
        :param outputs: Array of vector expected outputs for the network
        :return:        Results produced through the testing of the network
        """
        assert np.size(inputs[0, :]) == np.size(outputs[0, :])
        self.__mean_true_positives = 0.0
        self.__mean_true_negatives = 0.0
        self.__mean_false_negatives = 0.0
        self.__mean_false_positives = 0.0
        self.__mean_absolute_error = 0.0
        self.__mean_squared_error = 0.0
        results = np.zeros((np.size(outputs[:, 0]), np.size(outputs[0, :])))
        for i in range(np.size(inputs[0, :])):
            results[:, i] = self.forward_propagate(inputs[:, i])
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

        return results

    def get_accuracy(self) -> float:
        """
        Calculates the network's prediction accuracy from it's last metrics
        generation
        :return: Accuracy of the network's predictions
        """
        return 0.0 if (self.__mean_true_positives + self.__mean_true_negatives) == 0.0 else (
                (self.__mean_true_positives + self.__mean_true_negatives) / (
                 self.__mean_true_positives +
                 self.__mean_true_negatives +
                 self.__mean_false_positives +
                 self.__mean_false_negatives))

    def get_precision(self) -> float:
        """
        Calculates the network's prediction precision from it's last metrics
        generation
        :return: Precision of the network's predictions
        """
        return 0.0 if self.__mean_true_positives == 0.0 else (
                self.__mean_true_positives / (self.__mean_true_positives +
                                              self.__mean_false_positives))

    def get_recall(self) -> float:
        """
        Calculates the network's prediction recall from it's last metrics
        generation
        :return: Recall of the network's predictions
        """
        return 0.0 if self.__mean_true_positives == 0.0 else (
                self.__mean_true_positives / (self.__mean_true_positives +
                                              self.__mean_false_negatives))

    def get_specificity(self) -> float:
        """
        Calculates the network's prediction specificity from it's last metrics
        generation
        :return: Specificity of the network's predictions
        """
        return 0.0 if self.__mean_true_negatives == 0.0 else (
                self.__mean_true_negatives / (self.__mean_true_negatives +
                                              self.__mean_false_positives))

    def get_f1(self) -> float:
        """
        Calculates the network's prediction F1 from it's last metrics
        generation
        :return: F1 of the network's predictions
        """
        p = self.get_precision()
        r = self.get_recall()
        return 2 * r * p / (r + p)

    def get_mean_error(self) -> float:
        """
        Calculates the network's prediction mean absolute error from it's
        last metrics generation
        :return: Mean absolute error of the network's predictions
        """
        return self.__mean_absolute_error

    def get_mean_squared_error(self) -> float:
        """
        Calculates the network's prediction mean squared error from it's
        last metrics generation
        :return: Mean squared error of the network's predictions
        """
        return self.__mean_squared_error


def main(training_points_amount: int = 2000, testing_points_amount: int = 1000, training_epochs: int = 1000) -> None:
    """
    Generates plotted evidence of the network's learning capacity for a linear
    classification problem
    :param training_points_amount:  Amount of points to use for training
    :param testing_points_amount:   Amount of points to use for testing metrics
    :param training_epochs:         Amount of training epochs to run
    """
    training_plot = False
    testing_plot = False
    progress_plot = True

    network = Network(2, [3, 5, 3], 1, 0.025)

    slope = 2.0
    y_intercept = -1.0

    one_array = np.array([1.0])
    zero_array = np.array([0.0])

    line = np.zeros((2, 200))

    for i in range(200):
        line[0, i] = i - 100
        line[1, i] = y_intercept + slope * line[0, i]

    for n in range(training_epochs):

        training_points = np.random.uniform(-100 * int(slope), 100 * int(slope), (2, training_points_amount))
        testing_points = np.random.uniform(-100 * int(slope), 100 * int(slope), (2, testing_points_amount))

        # Training Classes

        if training_plot:

            _, ax = plt.subplots()

            ax.plot(line[0, :], line[1, :])

            training_classes = np.zeros((1, training_points_amount))
            for i, c in enumerate(training_classes[0, :]):
                training_classes[0, i] = one_array \
                    if training_points[1, i] > \
                    y_intercept + slope * training_points[0, i] \
                    else zero_array
                ax.scatter(training_points[0, i], training_points[1, i],
                           color='b' if training_classes[0, i] == one_array else 'r')

            plt.title("Training Reference for Epoch " + str(n))
            ax.grid(True)
            plt.show()

        else:

            training_classes = np.zeros((1, training_points_amount))
            for i, c in enumerate(training_classes[0, :]):
                training_classes[0, i] = one_array \
                    if training_points[1, i] > \
                    y_intercept + slope * training_points[0, i] \
                    else zero_array

        # Testing Classes

        if testing_plot:

            _, ax = plt.subplots()

            ax.plot(line[0, :], line[1, :])

            testing_classes = np.zeros((1, testing_points_amount))
            for i, c in enumerate(testing_classes[0, :]):
                testing_classes[0, i] = one_array \
                    if testing_points[1, i] > \
                    y_intercept + slope * testing_points[0, i] \
                    else zero_array
                ax.scatter(testing_points[0, i], testing_points[1, i],
                           color='b' if testing_classes[0, i] == one_array else 'r')

            plt.title("Testing Reference for Epoch " + str(n))
            ax.grid(True)
            plt.show()

        else:

            testing_classes = np.zeros((1, testing_points_amount))
            for i, c in enumerate(testing_classes[0, :]):
                testing_classes[0, i] = one_array \
                    if testing_points[1, i] > \
                    y_intercept + slope * testing_points[0, i] \
                    else zero_array

        # Training Results

        if progress_plot:

            _, ax = plt.subplots()

            ax.plot(line[0, :], line[1, :])

            network.train_epoch(training_points, training_classes)
            result = network.generate_metrics_epoch(testing_points, testing_classes)

            for i in range(testing_points_amount):
                ax.scatter(testing_points[0, i], testing_points[1, i],
                           color='b' if result[0, i] == 1.0 else 'r',
                           label='correct' if result[0, i] == testing_classes[0, i] else 'wrong')

            a = network.get_accuracy()
            p = network.get_precision()
            r = network.get_recall()
            s = network.get_specificity()
            ae = network.get_mean_error()
            se = network.get_mean_squared_error()

            plt.title("Training Result for Epoch " + str(n))
            ax.text(25, -150,
                    ("Accuracy for epoch " + str(n + 1) +
                     " equals                  = " + str(a) +
                     "\nPrecision for epoch " + str(n + 1) +
                     " equals                 = " + str(p) +
                     "\nRecall for epoch " + str(n + 1) +
                     " equals                     = " + str(r) +
                     "\nSpecificity for epoch " + str(n + 1) +
                     " equals               = " + str(s) +
                     "\nMean Error for epoch " + str(n + 1) +
                     " equals               = " + str(ae) +
                     "\nMean Squared Error for epoch " + str(n + 1) +
                     " equals = " + str(se)),
                    fontsize=5,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
            ax.grid(True)
            plt.show()

        else:

            network.train_epoch(training_points, training_classes)
            network.generate_metrics_epoch(testing_points, testing_classes)

            a = network.get_accuracy()
            p = network.get_precision()
            r = network.get_recall()
            s = network.get_specificity()
            ae = network.get_mean_error()
            se = network.get_mean_squared_error()

            print("---------------------------------------------------------")
            print("Accuracy for epoch ", n + 1, " equals           = ", a)
            print("Precision for epoch ", n + 1, " equals          = ", p)
            print("Recall for epoch ", n + 1, " equals             = ", r)
            print("Specificity for epoch ", n + 1, " equals        = ", s)
            print("Mean Error for epoch ", n + 1, " equals         = ", ae)
            print("Mean Squared Error for epoch ", n + 1, " equals = ", se)
            print("---------------------------------------------------------")


if __name__ == '__main__':
    main()
