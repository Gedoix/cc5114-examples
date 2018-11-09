import math
from unittest import TestCase

import numpy as np

from src.sigmoid_networks.network import Perceptron


class TestPerceptron(TestCase):

    def test_forward_propagate(self):
        p = Perceptron(2)
        ones = 0
        zeros = 0
        for i in range(1000):
            x = 1000000.0*np.cos(i/(2.0*math.pi))
            y = 1000000.0*np.sin(i/(2.0*math.pi))
            if p.forward_propagate(np.array([x, y])) >= 0.5:
                ones += 1
            else:
                zeros += 1
        self.assertAlmostEqual(ones, zeros, delta=20)

    def test_back_propagate_output_layer(self):
        p = Perceptron(2).custom(np.array([1.0, 1.0]), 1.0)
        p.forward_propagate(np.array([1.0, 0.0]))
        p.back_propagate_output_layer(0.1)
        self.assertAlmostEqual(0.01049935, p.get_delta(), delta=0.00000001)

    def test_back_propagate_from_weight(self):
        p = Perceptron(2).custom(np.array([1.0, 1.0]), 1.0)
        p.forward_propagate(np.array([1.0, 0.0]))
        p.back_propagate_from_weight(np.array(10**-0.5), np.array(10**-0.5))
        self.assertAlmostEqual(0.01049935, p.get_delta(), delta=0.00000001)

    def test_update_values(self):
        p = Perceptron(2).custom(np.array([1.0, 1.0]), 1.0)
        p.forward_propagate(np.array([1.0, 0.0]))
        p.back_propagate_from_weight(np.array(10 ** -0.5), np.array(10 ** -0.5))
        p.update_values()
        weights, bias = p.get_values()
        self.assertAlmostEqual(1.001049935, weights[0], delta=0.00000001)
        self.assertAlmostEqual(1.0, weights[1], delta=0.00000001)
        self.assertAlmostEqual(1.001049935, bias, delta=0.00000001)

    def test_get_delta_and_weight(self):
        p = Perceptron(2).custom(np.array([1.0, 1.0]), 1.0)
        p.forward_propagate(np.array([1.0, 0.0]))
        p.back_propagate_from_weight(np.array(10 ** -0.5), np.array(10 ** -0.5))
        p.update_values()
        delta, weight = p.get_delta_and_weight(0)
        self.assertAlmostEqual(0.01049935, delta, delta=0.00000001)
        self.assertAlmostEqual(1.001049935, weight, delta=0.00000001)
        delta, weight = p.get_delta_and_weight(1)
        self.assertAlmostEqual(0.01049935, delta, delta=0.00000001)
        self.assertAlmostEqual(1.0, weight, delta=0.00000001)
