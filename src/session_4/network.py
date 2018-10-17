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
        self.__last_feed = math.exp(-np.logaddexp(0, -np.dot(inputs, self.__weights) + self.__bias))
        return self.__last_feed

    def back_propagate(self, deltas: np.ndarray):
        error = np.dot(self.__weights, deltas)
        self.__delta = error * (self.__last_feed * (1.0 - self.__last_feed))
        self.__weights += self.__learning_rate * self.__delta * self.__last_inputs
        self.__bias += self.__learning_rate * self.__delta

    def get_delta(self):
        return self.__delta


class Layer:

    def __init__(self, input_amount: int, perceptron_amount: int, learning_rate: float = 0.1):
        self.__perceptrons = []
        while perceptron_amount > 0:
            self.__perceptrons.append(Perceptron(input_amount, learning_rate=learning_rate))
            perceptron_amount -= 1
        self.__last_inputs = None
        self.__last_feed = None
        self.__deltas = None

    def forward_propagate(self, inputs: np.ndarray):
        self.__last_inputs = inputs
        self.__last_feed = []
        for i in range(len(self.__perceptrons)):
            self.__last_feed.append(self.__perceptrons[i].forward_propagate(inputs))
        self.__last_feed = np.array(self.__last_feed)
        return np.array(self.__last_feed)

    def back_propagate(self, deltas: np.ndarray):
        for i in range(len(self.__perceptrons)):
            self.__perceptrons[i].back_propagate(deltas)

    def get_deltas(self):
        self.__deltas = []
        for i in range(len(self.__perceptrons)):
            self.__deltas.append(self.__perceptrons[i].get_delta())
        self.__deltas = np.array(self.__deltas)
        return self.__deltas


class Network:

    def __init__(self, input_amount: int, perceptron_per_hidden_layer_amounts: list, output_amount: int, learning_rate: float = 0.1):
        self.__hidden_layers = [Layer(input_amount, perceptron_per_hidden_layer_amounts[0], learning_rate=learning_rate)]
        for i in range(len(perceptron_per_hidden_layer_amounts)-1):
            self.__hidden_layers.append(Layer(perceptron_per_hidden_layer_amounts[i], perceptron_per_hidden_layer_amounts[i+1], learning_rate=learning_rate))
        self.__output_layer = Layer(perceptron_per_hidden_layer_amounts[-1], output_amount, learning_rate=learning_rate)
        self.__results = None

    def forward_propagate(self, inputs: np.ndarray):
        self.__results = self.__hidden_layers[0].forward_propagate(inputs)
        for i in range(len(self.__hidden_layers)-1):
            self.__results = self.__hidden_layers[i+1].forward_propagate(self.__results)
        self.__results = self.__output_layer.forward_propagate(self.__results)
        return self.__results

    def back_propagare(self, expected_outputs: np.ndarray):
        deltas = expected_outputs-self.__results
        self.__output_layer.back_propagate(deltas)
        deltas = self.__output_layer.get_deltas()
        for i in range(len(self.__hidden_layers)):
            j = len(self.__hidden_layers)-i-1
            self.__hidden_layers[j].back_propagate(deltas)
            deltas = self.__hidden_layers[j].get_deltas()

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        self.forward_propagate(inputs)
        self.back_propagare(outputs)
