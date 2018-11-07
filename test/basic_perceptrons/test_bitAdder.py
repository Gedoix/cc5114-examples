from unittest import TestCase

from basic_perceptrons.basic_networks import BitAdder, np


class TestBitAdder(TestCase):

    def test_add(self):
        n = BitAdder()

        if not np.array_equal(np.array([0, 0]), n.feed(np.array([0, 0]))):
            self.fail()
        if not np.array_equal(np.array([1, 0]), n.feed(np.array([1, 0]))):
            self.fail()
        if not np.array_equal(np.array([1, 0]), n.feed(np.array([0, 1]))):
            self.fail()
        if not np.array_equal(np.array([0, 1]), n.feed(np.array([1, 1]))):
            self.fail()
