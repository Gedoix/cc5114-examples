import numpy as np


class MatrixNetwork:

    def __init__(self, inputs_amount: int, hidden_layer_sizes: list, outputs_amount: int, learning_rate: float = 0.1):
        """
        Constructor for a perceptron network
        :param inputs_amount:                        Amount of inputs the
        network will accept
        :param hidden_layer_sizes: List of amounts of
        perceptrons per layer
        :param outputs_amount:                       Length of the output vector
        of the network
        :param learning_rate:                       Learning rate for all
        perceptrons in the network
        """

        assert(inputs_amount > 0)
        assert(outputs_amount > 0)
        for size in hidden_layer_sizes:
            assert(isinstance(size, int))
            assert(size > 0)

        self.__inputs_amount = inputs_amount
        self.__outputs_amount = outputs_amount
        self.__layers = []
        self.__learning_rate = learning_rate
        self.__results = None

        if len(hidden_layer_sizes) == 0:
            self.__layers.append(np.random.uniform(-2.0, 2.0, (inputs_amount, outputs_amount)))

        else:
            self.__layers.append(np.random.uniform(-2.0, 2.0, (inputs_amount, hidden_layer_sizes[0])))
            for i in range(len(hidden_layer_sizes) - 1):
                self.__layers.append(np.random.uniform(-2.0, 2.0, (hidden_layer_sizes[i], hidden_layer_sizes[i + 1])))
            self.__layers.append(np.random.uniform(-2.0, 2.0, (hidden_layer_sizes[len(hidden_layer_sizes) - 1],
                                                               outputs_amount)))

        self.__mean_true_positives = 0.0
        self.__mean_true_negatives = 0.0
        self.__mean_false_negatives = 0.0
        self.__mean_false_positives = 0.0
        self.__mean_absolute_error = 0.0
        self.__mean_squared_error = 0.0

    def forward_propagate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Produces the output of the network by evaluating all layers within
        :param inputs:  Input vector of the network
        :return:        Output vector of the network
        """
        assert(len(inputs) == self.__inputs_amount)

        self.__results = []

        # The results propagate through the first layer
        self.__results.append(np.array(inputs))

        # The results propagate through the rest of the layers
        for layer in self.__layers:
            self.__results.append(np.multiply(self.__results[len(self.__results) - 1], layer))

        # The results of the last layer are extracted
        return self.__results[len(self.__results) - 1]

    def back_propagate(self, expected_outputs: np.ndarray) -> None:
        """
        Propagates the network's error derivative through all layers
        :param expected_outputs:    Output expected for the given input
        """
        assert (len(expected_outputs) == self.__outputs_amount)

        pass

    def update_values(self) -> None:
        """
        Updates internal values within the network
        """
        pass

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
