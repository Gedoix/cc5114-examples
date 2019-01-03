from neuroevolution_algorithms.neuron import GeneticNeuron


class GeneticSynapse:

    __availability: bool
    __output: float
    __end_neuron: GeneticNeuron
    __weight: float
    __start_neuron: GeneticNeuron

    # Constructor

    def __init__(self, start_neuron: GeneticNeuron, weight: float, end_neuron: GeneticNeuron):
        self.__start_neuron = start_neuron
        self.__weight = weight
        self.__end_neuron = end_neuron
        self.__output = 0.0
        self.__availability = True

    # Internal value management

    def set_output(self, result: float) -> None:
        self.__output = self.__weight*result

    def get_output(self) -> float:
        if self.__start_neuron.get_calculations() < self.__end_neuron.get_calculations():
            self.__start_neuron.calculate()
        return self.__output

    # Availability management

    def enable(self) -> None:
        self.__availability = True

    def disable(self) -> None:
        self.__availability = False

    def is_available(self) -> bool:
        return self.__availability

    # SETTERS

    def set_start(self, start_neuron: GeneticNeuron) -> None:
        self.__start_neuron = start_neuron

    def set_weight(self, weight: float) -> None:
        self.__weight = weight

    def set_end(self, end_neuron: GeneticNeuron) -> None:
        self.__end_neuron = end_neuron

    # GETTERS

    def get_start(self) -> GeneticNeuron:
        return self.__start_neuron

    def get_weight(self) -> float:
        return self.__weight

    def get_end(self) -> GeneticNeuron:
        return self.__end_neuron
