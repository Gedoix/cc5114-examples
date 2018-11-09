from src.basic_perceptrons.basic_perceptrons import *


class BasicNetwork:

    def __init__(self, perceptron_amount: int, synapses: dict,
                 base_perceptrons: dict = None):
        """
        Initializer for the network
        synapses format:    {p3:[p1, p2], p2:[p1]} describes a network where
                                p1 feeds from p1 and p2, and p2 feeds from p1
                            negative numbers and 0 denote input indices
                            positive numbers denote perceptrons
        :param perceptron_amount:   Total amount of perceptrons allowed
        :param synapses:            Mapping of directed synapses between perceptrons within the network
        :param base_perceptrons:    Base perceptrons to use when generating the network
        """
        self.__synapses = synapses
        self.__perceptrons = {}
        while perceptron_amount != 0:
            if base_perceptrons is not None \
                    and base_perceptrons[perceptron_amount] is not None:
                self.__perceptrons[perceptron_amount] = \
                    base_perceptrons[perceptron_amount]
            else:
                self.__perceptrons[perceptron_amount] = Perceptron
                self.__perceptrons[perceptron_amount].\
                    set_values(weights_amount=len(self.__synapses[perceptron_amount]))
            perceptron_amount -= 1
        self.__outputs = []
        for perceptron1 in self.__synapses.keys():
            is_output = True
            for perceptron2 in self.__synapses.keys():
                if perceptron1 in self.__synapses[perceptron2]:
                    is_output = False
            if is_output:
                self.__outputs.append(perceptron1)

    def feed(self, inputs: np.ndarray):
        """
        Calculates the network's output by forward-propagating values through the perceptrons
        :param inputs:  Binary inputs for the network
        :return:        Binary output
        """
        activations = {}
        for perceptron in self.__synapses.keys():
            specific_inputs = []
            for specific_input in self.__synapses[perceptron]:
                if -specific_input in range(len(inputs)):
                    specific_inputs.append(inputs[-specific_input])
                else:
                    specific_inputs.append(activations[specific_input])
            activations[perceptron] = self.__perceptrons[perceptron].\
                feed(np.array(specific_inputs))
        results = []
        for output in self.__outputs:
            results.append(activations[output])
        return np.array(results)


class BitAdder(BasicNetwork):

    def __init__(self):
        """
        Initializer for a 2-input and 2-output binary bit adder network
        """
        super().__init__(5,
                         {1: [-1, 0],
                          2: [0, 1],
                          3: [-1, 1],
                          4: [2, 3],
                          5: [1, 1]},
                         base_perceptrons={1: NAND(),
                                           2: NAND(),
                                           3: NAND(),
                                           4: NAND(),
                                           5: NAND()})
