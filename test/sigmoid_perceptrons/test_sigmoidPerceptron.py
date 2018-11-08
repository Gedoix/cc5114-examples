from unittest import TestCase

import numpy as np

from sigmoid_perceptrons.sigmoid_perceptrons import SigmoidPerceptron


class TestSigmoidPerceptron(TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(0.5, SigmoidPerceptron.sigmoid(0), delta=0.0001)
        self.assertAlmostEqual(0.7310585786300049, SigmoidPerceptron.sigmoid(1), delta=0.0001)
        self.assertAlmostEqual(0.2689414213699951, SigmoidPerceptron.sigmoid(-1), delta=0.0001)
        self.assertAlmostEqual(0.8807970779778824, SigmoidPerceptron.sigmoid(2), delta=0.0001)
        self.assertAlmostEqual(0.11920292202211753, SigmoidPerceptron.sigmoid(-2), delta=0.0001)

    def test_feed(self):
        p = SigmoidPerceptron().built_with(weights_amount=2)
        self.assertIsInstance(p.feed(np.array([1.0, 1.0])), float)
        p = SigmoidPerceptron().built_with(weights_amount=3)
        self.assertIsInstance(p.feed(np.array([1.0, 1.0, 1.0])), float)
        p = SigmoidPerceptron().built_with(weights_amount=10)
        self.assertIsInstance(p.feed(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])), float)
