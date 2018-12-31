from neuroevolution_algorithms.neuron import Neuron


class GeneticSynapse:

    def __init__(self, start_neuron: Neuron, weight: float, end_neuron: Neuron):
        self.__start_neuron = start_neuron
        self.__weight = weight
        self.__end_neuron = end_neuron
        self.__output = 0.0

    def __copy__(self) -> "GeneticSynapse":
        return GeneticSynapse(self.__start_neuron, self.__weight, self.__end_neuron)

    def set_output(self, result: [bool, float]) -> None:
        self.__output = self.__weight*float(result)

    def get_output(self) -> float:
        return self.__output

    def set_start(self, start_neuron: Neuron) -> None:
        self.__start_neuron = start_neuron

    def set_weight(self, weight: float) -> None:
        self.__weight = weight

    def set_end(self, end_neuron: Neuron) -> None:
        self.__end_neuron = end_neuron

    def get_start(self) -> Neuron:
        return self.__start_neuron

    def get_weight(self) -> float:
        return self.__weight

    def get_end(self) -> Neuron:
        return self.__end_neuron
