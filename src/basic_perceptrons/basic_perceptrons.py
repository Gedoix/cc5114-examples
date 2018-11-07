import random

import numpy as np


class Perceptron:

    def __init__(self):
        """
        Initializer for the perceptron, sets all parameters to null default values
        """
        self.__bias = 0.0
        self.__weights = np.array([])
        self.__learning_rate = 0.0

    def set_values(self, bias: float = None, weights_amount: int = None,
                   weights: np.ndarray = None, learning_rate: float = None):
        """
        Setter for the Perceptron's parameters
        :param bias:            Activation bias
        :param weights_amount:  Amount of weights to generate at random, one for each input
        :param weights:         Array of pre-made weights
        :param learning_rate:   Learning speed, a high value may endanger chances of convergence
        """
        self.__bias = random.uniform(-1.0, 1.0) if bias is None \
            else bias
        weights_amount = 2 if weights_amount is None \
            else weights_amount
        self.__weights = 2 * np.random.random(weights_amount) - 1.0 \
            if (weights is None) \
            else weights
        self.__learning_rate = 0.1 if learning_rate is None \
            else learning_rate

    def built_with(self, bias: float = None, weights_amount: int = None,
                   weights: np.ndarray = None, learning_rate: float = None):
        """
        Builder-like method for use when inline initializing a Perceptron
        :param bias:            Activation bias
        :param weights_amount:  Amount of weights to generate at random, one for each input
        :param weights:         Array of pre-made weights
        :param learning_rate:   Learning speed, a high value may endanger chances of convergence
        :return:                Initialized Perceptron
        """
        self.set_values(bias=bias, weights_amount=weights_amount,
                        weights=weights, learning_rate=learning_rate)
        return self

    def evaluate(self, inputs: np.ndarray):
        """
        Calculates the raw output of weighting inputs
        :param inputs:  Binary inputs
        :return:        Float raw output
        """
        return np.dot(inputs, self.__weights) + self.__bias

    def feed(self, inputs: np.ndarray):
        """
        Calculates the real binary output of weighting inputs
        :param inputs:  Binary inputs
        :return:        Binary output
        """
        return 1 \
            if self.evaluate(inputs) > 0 \
            else 0

    def learn(self, output_expected: int, inputs: np.ndarray, times: int = 1):
        """
        Trains through a single neuron back-propagation
        :param output_expected:     Expected output, for error calculation
        :param inputs:              Inputs producing said output
        :param times:               Amount of times to train
        """
        while times != 0:
            output_actual = self.feed(inputs)
            step = ((output_expected - output_actual) * self.__learning_rate)
            self.__weights += inputs * step
            self.__bias += step
            times -= 1


class AND(Perceptron):

    def __init__(self):
        """
        Initializer for the and gate, sets all parameters to default values
        """
        super().__init__()
        self.set_values(bias=-3.0, weights=np.array([2.0, 2.0]))


class OR(Perceptron):

    def __init__(self):
        """
        Initializer for the or gate, sets all parameters to default values
        """
        super().__init__()
        self.set_values(bias=0.0, weights=np.array([1.0, 1.0]))


class NAND(Perceptron):

    def __init__(self):
        """
        Initializer for the nand gate, sets all parameters to default values
        """
        super().__init__()
        self.set_values(bias=3.0, weights=np.array([-2.0, -2.0]))
