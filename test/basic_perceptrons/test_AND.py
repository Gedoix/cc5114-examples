from unittest import TestCase

import numpy as np

from src.basic_perceptrons.basic_perceptrons import AND


class TestAND(TestCase):

    def test_feed(self):
        p = AND()

        if 0 != p.feed(np.array([0, 0])):
            self.fail()
        if 0 != p.feed(np.array([1, 0])):
            self.fail()
        if 0 != p.feed(np.array([0, 1])):
            self.fail()
        if 1 != p.feed(np.array([1, 1])):
            self.fail()
