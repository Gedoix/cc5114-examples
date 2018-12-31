from neuroevolution_algorithms.neuron import InputNeuron


class Network:

    def __init__(self, input_amount: int, output_amount: int):
        self.inputs = []
        self.outputs = []
        for _ in range(input_amount):
            self.inputs.append(InputNeuron())
        for _ in range(output_amount):
            self.outputs.append(OutputNeuron())

