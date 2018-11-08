from unittest import TestCase

import numpy as np

from sigmoid_perceptrons.sigmoid_perceptrons import SigmoidClassifier


class TestSigmoidClassifier(TestCase):

    def test_classification(self):
        # Testing an arbitrary line is being drawn, wherever that may be, with two points somewhat far away from the
        # random values the slope and y_intercept can interfere with
        c = SigmoidClassifier(1.0, 0.0)
        classification = c.expected_classification(1.0, 2.0)
        self.assertEqual(1 if classification == 0 else 0, c.expected_classification(-1.0, -2.0))

    def test_train(self):
        c = SigmoidClassifier(2.5871929, 898.7397737492184637)
        accuracy_1 = 0
        for i in range(1):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            c.train(x, y, c.expected_classification(x, y))
        for i in range(1000):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            result = c.classification(x, y)
            accuracy_1 += 1 if result == c.expected_classification(x, y) else 0
        accuracy_1 /= 1000.0

        accuracy_2 = 0
        for i in range(5000):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            c.train(x, y, c.expected_classification(x, y))
        for i in range(1000):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            result = c.classification(x, y)
            accuracy_2 += 1 if result == c.expected_classification(x, y) else 0
        accuracy_2 /= 1000.0

        self.assertGreaterEqual(accuracy_2, accuracy_1)

    def test_auto_train(self):
        c = SigmoidClassifier(2.5871929, 898.7397737492184637)
        accuracy_1 = 0
        c.auto_train(1)
        for i in range(1000):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            result = c.classification(x, y)
            accuracy_1 += 1 if result == c.expected_classification(x, y) else 0
        accuracy_1 /= 1000.0

        accuracy_2 = 0
        c.auto_train(5000)
        for i in range(1000):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            result = c.classification(x, y)
            accuracy_2 += 1 if result == c.expected_classification(x, y) else 0
        accuracy_2 /= 1000.0

        self.assertGreaterEqual(accuracy_2, accuracy_1)
