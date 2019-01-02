from typing import List

from neuroevolution_algorithms.synapse import GeneticSynapse


class GeneticNeuron:
    
    __calculations: int

    # Constructor

    def __init__(self):
        self.__calculations = 0

    # Object cloning

    def clone(self) -> "GeneticNeuron":
        new = GeneticNeuron()
        return new

    # Calculations counter

    def set_calculations(self, calculations: int) -> None:
        self.__calculations = calculations
        
    def get_calculations(self) -> int:
        return self.__calculations

    # Main behaviour

    def calculate(self) -> None:
        self.__calculations += 1
        pass


class InputNeuron(GeneticNeuron):

    __value: float
    __outputs: List[GeneticSynapse]

    def __init__(self):
        super().__init__()
        self.__outputs = []
        self.__value = 0.0

    def clone(self) -> "InputNeuron":
        new = InputNeuron()
        new.__value = self.__value
        return new

    def set_value(self, value: float) -> None:
        self.__value = value

    def add_output(self, synapse: GeneticSynapse) -> None:
        self.__outputs.append(synapse)

    def calculate(self):
        super().calculate()
        for o in self.__outputs:
            o.set_output(self.__value)


class HiddenNeuron(GeneticNeuron):

    __outputs: List[GeneticSynapse]
    __inputs: List[GeneticSynapse]
    __bias: float

    def __init__(self, bias: float):
        super().__init__()
        self.__bias = bias
        self.__inputs = []
        self.__outputs = []

    def clone(self) -> "HiddenNeuron":
        new = HiddenNeuron(self.__bias)
        new.__bias = self.__bias
        return new

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def add_output(self, synapse: GeneticSynapse) -> None:
        self.__outputs.append(synapse)

    def calculate(self) -> None:
        super().calculate()
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        last_result = weighted_sum + self.__bias >= 0
        for o in self.__outputs:
            o.set_output(last_result)

    def set_bias(self, bias: float) -> None:
        self.__bias = bias


class OutputNeuron(GeneticNeuron):

    __inputs: List[GeneticSynapse]
    __result: bool

    def __init__(self):
        super().__init__()
        self.__inputs = []
        self.__result = False

    def clone(self) -> "OutputNeuron":
        new = OutputNeuron()
        new.__result = self.__result
        return new

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def calculate(self) -> None:
        super().calculate()
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        self.__result = weighted_sum >= 0

    def get_result(self) -> bool:
        return self.__result
