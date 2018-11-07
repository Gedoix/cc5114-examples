import numpy as np

from session_1.basic_perceptrons import Perceptron
from session_3.sigmoid_perceptrons import SigmoidPerceptron

class logic_trainer:

    def __init__(self):
        self.inputs = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
        self.outputs = {"and": [0, 0, 0, 1], "or": [0, 1, 1, 1], "nand": [1, 1, 1, 0], "xor": [0, 1, 1, 0]}
        self.learners = {"and": [Perceptron().built_with(weights_amount=2), SigmoidPerceptron().built_with(weights_amount=2)],
                        "or": [Perceptron().built_with(weights_amount=2), SigmoidPerceptron().built_with(weights_amount=2)],
                        "nand": [Perceptron().built_with(weights_amount=2), SigmoidPerceptron().built_with(weights_amount=2)],
                        "xor": [Perceptron().built_with(weights_amount=2), SigmoidPerceptron().built_with(weights_amount=2)]}

    def get_accuracy(self):
        results = np.zeros(8)
        index = 0
        for gate in self.outputs.keys():
            for learner in self.learners.get(gate):
                for i in range(4):
                    results[index] += 1 if learner.feed(self.inputs[i]) == self.outputs.get(gate)[i] else 0
                results[index] *= 25
                index += 1
        return results
