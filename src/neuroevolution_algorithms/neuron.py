from neuroevolution_algorithms.synapse import GeneticSynapse


class Neuron:
    pass


class InputNeuron(Neuron):

    def __init__(self):
        self.__outputs = []

    def __copy__(self) -> "InputNeuron":
        new = InputNeuron()
        for o in self.__outputs[:]:
            new.add_output(o)
        return new

    def add_output(self, synapse: GeneticSynapse) -> None:
        self.__outputs.append(synapse)

    def calculate(self, i: float):
        for o in self.__outputs:
            o.set_output(i)


class GeneticNeuron(Neuron):

    def __init__(self, bias: float):
        self.__bias = bias
        self.__inputs = []
        self.__outputs = []

    def __copy__(self) -> "GeneticNeuron":
        new = GeneticNeuron(self.__bias)
        for i in self.__inputs[:]:
            new.add_input(i)
        for o in self.__outputs[:]:
            new.add_output(o)
        return new

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def add_output(self, synapse: GeneticSynapse) -> None:
        self.__outputs.append(synapse)

    def calculate(self) -> None:
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        last_result = weighted_sum + self.__bias >= 0
        for o in self.__outputs:
            o.set_output(last_result)

    def set_bias(self, bias: float) -> None:
        self.__bias = bias


class OutputNeuron(Neuron):

    def __init__(self, bias: float):
        self.__bias = bias
        self.__inputs = []
        self.__result = False

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def calculate(self) -> None:
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        self.__result = weighted_sum + self.__bias >= 0

    def get_result(self) -> bool:
        return self.__result
