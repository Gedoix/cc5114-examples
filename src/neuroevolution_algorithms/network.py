import random
from typing import List, Any, Tuple

from neuroevolution_algorithms.neuron import InputNeuron, HiddenNeuron, OutputNeuron
from neuroevolution_algorithms.synapse import GeneticSynapse


def find(l: List[Any], obj: Any) -> int:
    try:
        return l.index(obj)
    except ValueError:
        return -1


class Network:

    __shared_fitness: float
    __fitness: float
    __birth_generation: int
    __synapses: List[GeneticSynapse]
    __input_neurons: List[InputNeuron]
    __hidden_neurons: List[HiddenNeuron]
    __output_neurons: List[OutputNeuron]
    __generator: random.Random

    def __init__(self, input_amount: int, output_amount: int, seed: int = None, birth_generation: int = 1):
        self.__synapses = []
        self.__input_neurons = []
        self.__hidden_neurons = []
        self.__output_neurons = []
        for _ in range(input_amount):
            self.__input_neurons.append(InputNeuron())
        for _ in range(output_amount):
            self.__output_neurons.append(OutputNeuron())
        for i in self.__input_neurons:
            for o in self.__output_neurons:
                self.__synapses.append(GeneticSynapse(i, self.__generator.normalvariate(0, 5), o))
        self.__generator = random.Random()
        if seed is not None:
            self.__generator.seed(seed)
        self.__birth_generation = birth_generation
        self.__fitness = 0.0
        self.__shared_fitness = 0.0

    def clone(self, seed: int = None, birth_generation: int = None) -> "Network":
        if birth_generation is not None:
            new = Network(len(self.__input_neurons), len(self.__output_neurons),
                          seed=seed, birth_generation=birth_generation)
        else:
            new = Network(len(self.__input_neurons), len(self.__output_neurons), seed=seed)
        for neuron in self.__hidden_neurons:
            new.__hidden_neurons.append(neuron.clone())
        for synapse in self.__synapses:
            index_1 = find(self.__input_neurons, synapse.get_start())
            if index_1 != -1:
                start_neuron = new.__input_neurons[index_1]
            else:
                index_1 = find(self.__hidden_neurons, synapse.get_start())
                start_neuron = new.__hidden_neurons[index_1]
            index_2 = find(self.__output_neurons, synapse.get_end())
            if index_2 != -1:
                end_neuron = new.__output_neurons[index_2]
            else:
                index_2 = find(self.__hidden_neurons, synapse.get_end())
                end_neuron = new.__hidden_neurons[index_2]
            new.__synapses.append(GeneticSynapse(start_neuron, synapse.get_weight(), end_neuron))
        new.__fitness = self.__fitness
        new.__shared_fitness = self.__shared_fitness
        return new

    def get_signature(self) -> Tuple[int, int]:
        return len(self.__input_neurons), len(self.__output_neurons)

    def get_innovation(self) -> int:
        return len(self.__synapses)

    def calculate(self, inputs: List[int, float]) -> None:
        if len(inputs) != len(self.__input_neurons):
            raise RuntimeError("Wrong input length")
        for index, i in enumerate(self.__input_neurons):
            i.set_value(inputs[index])
        for o in self.__output_neurons:
            o.calculate()

    def increase_generation(self) -> None:
        self.__birth_generation += 1

    def get_generation(self) -> int:
        return self.__birth_generation

    def mutation_add_synapse(self) -> None:
        i = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__input_neurons) - 1)
        j = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__output_neurons) - 1)
        if i < len(self.__hidden_neurons) and j < len(self.__hidden_neurons):
            while i >= j:
                i = self.__generator.randint(0, len(self.__hidden_neurons) - 1)
                j = self.__generator.randint(0, len(self.__hidden_neurons) - 1)
        if i < len(self.__hidden_neurons):
            start_neuron = self.__hidden_neurons[i]
        else:
            start_neuron = self.__input_neurons[i - len(self.__hidden_neurons)]

        if j < len(self.__hidden_neurons):
            end_neuron = self.__hidden_neurons[j]
        else:
            end_neuron = self.__output_neurons[j - len(self.__hidden_neurons)]

        weight = self.__generator.normalvariate(0, 5)
        self.__synapses.append(GeneticSynapse(start_neuron, weight, end_neuron))

    def mutation_add_neuron(self) -> None:
        old_synapse = self.__synapses[self.__generator.randint(0, len(self.__synapses) - 1)]
        old_synapse.disable()
        start_neuron = old_synapse.get_start()
        end_neuron = old_synapse.get_end()
        bias = self.__generator.normalvariate(0, 5)
        new_neuron = HiddenNeuron(bias)
        new_weight_1 = 1
        new_weight_2 = old_synapse.get_weight()
        self.__synapses.append(GeneticSynapse(start_neuron, new_weight_1, new_neuron))
        if isinstance(start_neuron, InputNeuron):
            self.__hidden_neurons.insert(0, new_neuron)
        elif isinstance(end_neuron, OutputNeuron):
            self.__hidden_neurons.append(new_neuron)
        else:
            index_1 = find(self.__hidden_neurons, start_neuron)
            index_2 = find(self.__hidden_neurons, end_neuron)
            self.__hidden_neurons. insert(int((index_1+index_2)/2.0 + 1), new_neuron)
        self.__synapses.append(GeneticSynapse(new_neuron, new_weight_2, end_neuron))

    def crossover(self, other: "Network") -> None:
        pass

    def set_seed(self, seed: int):
        self.__generator.seed(seed)

    def set_fitness(self, fitness: float) -> None:
        self.__fitness = fitness

    def set_shared_fitness(self, shared_fitness: float) -> None:
        self.__shared_fitness = shared_fitness

    def get_fitness(self) -> float:
        return self.__fitness

    def get_shared_fitness(self) -> float:
        return self.__shared_fitness
