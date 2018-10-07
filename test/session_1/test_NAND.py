from unittest import TestCase

import numpy as np

from session_1.basic_perceptrons import NAND


class TestNAND(TestCase):

    def test_feed(self):
        p = NAND()

        if 1 != p.feed(np.array([0, 0])):
            self.fail()
        if 1 != p.feed(np.array([1, 0])):
            self.fail()
        if 1 != p.feed(np.array([0, 1])):
            self.fail()
        if 0 != p.feed(np.array([1, 1])):
            self.fail()
