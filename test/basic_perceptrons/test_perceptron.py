from unittest import TestCase

from basic_perceptrons.basic_perceptrons import *


class TestPerceptron(TestCase):

    def test_feed(self):
        p1 = Perceptron()
        p1.set_values(bias=-2.0, weights=np.array([1.0, 2.0, 3.0]))
        p2 = Perceptron()
        p2.set_values(bias=0.0, weights=np.array([0.0, 0.0, 0.0]))
        p3 = Perceptron()
        p3.set_values(bias=-30.0, weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))

        if 0 != p1.feed(np.array([0.0, 0.0, 0.0])):
            self.fail()
        if 0 != p1.feed(np.array([1.0, 0.0, 0.0])):
            self.fail()
        if 0 != p1.feed(np.array([0.0, 1.0, 0.0])):
            self.fail()
        if 1 != p1.feed(np.array([0.0, 0.0, 1.0])):
            self.fail()
        if 1 != p1.feed(np.array([1.0, 1.0, 0.0])):
            self.fail()
        if 1 != p1.feed(np.array([1.0, 0.0, 1.0])):
            self.fail()
        if 1 != p1.feed(np.array([0.0, 1.0, 1.0])):
            self.fail()
        if 1 != p1.feed(np.array([1.0, 1.0, 1.0])):
            self.fail()

        if 0 != p2.feed(np.array([0.0, 0.0, 0.0])):
            self.fail()
        if 0 != p2.feed(np.array([1.0, 0.0, 0.0])):
            self.fail()
        if 0 != p2.feed(np.array([0.0, 1.0, 0.0])):
            self.fail()
        if 0 != p2.feed(np.array([0.0, 0.0, 1.0])):
            self.fail()
        if 0 != p2.feed(np.array([1.0, 1.0, 0.0])):
            self.fail()
        if 0 != p2.feed(np.array([1.0, 0.0, 1.0])):
            self.fail()
        if 0 != p2.feed(np.array([0.0, 1.0, 1.0])):
            self.fail()
        if 0 != p2.feed(np.array([1.0, 1.0, 1.0])):
            self.fail()

        if 0 != p3.feed(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
            self.fail()
        if 0 != p3.feed(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])):
            self.fail()
        if 1 != p3.feed(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])):
            self.fail()
