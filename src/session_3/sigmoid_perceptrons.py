import numpy as np

from session_1.basic_perceptrons import Perceptron


class SigmoidPerceptron(Perceptron):

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(z))

    def feed(self, inputs: np.ndarray):
        return self.sigmoid(self.evaluate(inputs))

def perceptron_logical_accuracies():
    


def sigmoid_logical_accuracies():



if __name__ == '__main__':
    perceptron_logical_acuracies()
    sigmoid_logical_acuracies()
