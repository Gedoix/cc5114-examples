from unittest import TestCase

from src.basic_perceptrons.basic_networks import BasicNetwork, Perceptron, np


class TestNetwork(TestCase):

    def test_feed(self):
        n = BasicNetwork(5,
                         {1: [0, -1, -2],
                     2: [0, -1, -2],
                     3: [0, -1, -2],
                     4: [1, 2, 3],
                     5: [1, 2, 3]},
                         {1: Perceptron().built_with(bias=-5, weights=np.array([2, 2, 2])),
                     2: Perceptron().built_with(bias=0, weights=np.array([2, 2, 2])),
                     3: Perceptron().built_with(bias=-5, weights=np.array([2, 2, 2])),
                     4: Perceptron().built_with(bias=-5, weights=np.array([2, 2, 2])),
                     5: Perceptron().built_with(bias=0, weights=np.array([2, 2, 2]))})

        if not np.array_equal(np.array([0, 0]), n.feed(np.array([0, 0, 0]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([1, 0, 0]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([0, 1, 0]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([0, 0, 1]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([1, 1, 0]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([1, 0, 1]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([0, 1, 1]))):
            self.fail()
        if not np.array_equal(np.array([1, 1]), n.feed(np.array([1, 1, 1]))):
            self.fail()
