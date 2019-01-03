from typing import List

from neuroevolution_algorithms.synapse import GeneticSynapse


class GeneticNeuron:
    
    __calculations: int

    # Constructor

    def __init__(self):
        self.__calculations = 0

    # Calculations counter

    def set_calculations(self, calculations: int) -> None:
        self.__calculations = calculations
        
    def get_calculations(self) -> int:
        return self.__calculations

    # Main behaviour

    def calculate(self) -> None:
        self.__calculations += 1
        pass

    @staticmethod
    def sigmoid(x: float) -> float:
        import numpy as np
        import math
        return math.exp(-np.logaddexp(0.0, -x))


class InputNeuron(GeneticNeuron):

    __value: float
    __outputs: List[GeneticSynapse]

    def __init__(self):
        super().__init__()
        self.__outputs = []
        self.__value = 0.0

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

    def __init__(self):
        super().__init__()
        self.__inputs = []
        self.__outputs = []

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def add_output(self, synapse: GeneticSynapse) -> None:
        self.__outputs.append(synapse)

    def calculate(self) -> None:
        super().calculate()
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        last_result = self.sigmoid(weighted_sum)
        for o in self.__outputs:
            o.set_output(last_result)


class OutputNeuron(GeneticNeuron):

    __inputs: List[GeneticSynapse]
    __result: bool

    def __init__(self):
        super().__init__()
        self.__inputs = []
        self.__result = False

    def add_input(self, synapse: GeneticSynapse) -> None:
        self.__inputs.append(synapse)

    def calculate(self) -> None:
        super().calculate()
        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.get_output()
        self.__result = self.sigmoid(weighted_sum) > 0.5

    def get_result(self) -> bool:
        return self.__result
