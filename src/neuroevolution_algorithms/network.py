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

    __symbolic_synapses: List[Tuple[int, bool, int, bool]]
    __shared_fitness: float
    __fitness: float
    __birth_generation: int
    __synapses: List[GeneticSynapse]
    __input_neurons: List[InputNeuron]
    __hidden_neurons: List[HiddenNeuron]
    __output_neurons: List[OutputNeuron]
    __generator: random.Random

    def __init__(self, input_amount: int, output_amount: int, seed: int = None, birth_generation: int = None):
        self.__synapses = []
        self.__symbolic_synapses = []
        self.__input_neurons = []
        self.__hidden_neurons = []
        self.__output_neurons = []
        self.__start_seed = seed
        self.__generator = random.Random()
        self.__birth_generation = 1
        self.__fitness = 0.0
        self.__shared_fitness = 0.0
        for _ in range(input_amount):
            self.__input_neurons.append(InputNeuron())
        for _ in range(output_amount):
            self.__output_neurons.append(OutputNeuron())
        for i_index, i in enumerate(self.__input_neurons):
            for o_index, o in enumerate(self.__output_neurons):
                self.__synapses.append(GeneticSynapse(i, self.get_random_float(), o))
                self.__symbolic_synapses.append((i_index, True, o_index, True))
        if seed is not None:
            self.__generator.seed(seed)
        if birth_generation is not None:
            self.__birth_generation = birth_generation

    def clone(self, birth_generation: int = None) -> "Network":
        new = Network(len(self.__input_neurons), len(self.__output_neurons),
                      seed=self.__start_seed, birth_generation=birth_generation)

        new.set_seed(self.get_seed())

        for _ in self.__hidden_neurons:
            new.__hidden_neurons.append(HiddenNeuron())

        for synapse_index, index_1, is_input, index_2, is_output in enumerate(self.__symbolic_synapses):
            if is_input:
                start_neuron = new.__input_neurons[index_1]
            else:
                start_neuron = new.__hidden_neurons[index_1]
            if is_output:
                end_neuron = new.__output_neurons[index_2]
            else:
                end_neuron = new.__hidden_neurons[index_2]

            new.__synapses.append(GeneticSynapse(start_neuron, self.__synapses[synapse_index].get_weight(), end_neuron))
            new.__symbolic_synapses.append((index_1, is_input, index_2, is_output))

        new.__fitness = self.__fitness
        new.__shared_fitness = self.__shared_fitness

        return new

    def calculate(self, inputs: List[int, float]) -> List[bool]:
        if len(inputs) != len(self.__input_neurons):
            raise RuntimeError("Wrong input length, doesn't correspond with the network's signature")

        for index, i in enumerate(self.__input_neurons):
            i.set_value(inputs[index])

        results = []
        for o in self.__output_neurons:
            o.calculate()
            results.append(o.get_result())

        return results

    def mutation_change_weights(self) -> None:
        for synapse in self.__synapses:
            if self.__generator.randint(1, 100) <= 90:
                perturbation = 1 if self.__generator.randint(1, 100) <= 50 else -1
                new_weight = synapse.get_weight() + perturbation
            else:
                new_weight = self.get_random_float()
            synapse.set_weight(new_weight)

    def mutation_add_synapse(self) -> None:
        i = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__input_neurons) - 1)
        o = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__output_neurons) - 1)

        is_input = not i < len(self.__hidden_neurons)
        is_output = not o < len(self.__hidden_neurons)

        if not (is_input or is_output):
            while i >= o:
                i = self.__generator.randint(0, len(self.__hidden_neurons) - 1)
                o = self.__generator.randint(0, len(self.__hidden_neurons) - 1)
        if is_input:
            i = i - len(self.__hidden_neurons)
            start_neuron = self.__input_neurons[i]
        else:
            start_neuron = self.__hidden_neurons[i]

        if is_output:
            o = o - len(self.__hidden_neurons)
            end_neuron = self.__output_neurons[o]
        else:
            end_neuron = self.__hidden_neurons[o]

        weight = self.__generator.normalvariate(0, 5)
        self.__synapses.append(GeneticSynapse(start_neuron, weight, end_neuron))
        self.__symbolic_synapses.append((i, is_input, o, is_output))

    def mutation_add_neuron(self) -> None:
        old_synapse_index = self.__generator.randint(0, len(self.__synapses) - 1)
        old_synapse = self.__synapses[old_synapse_index]
        old_index_1, old_is_input, old_index_2, old_is_output = self.__symbolic_synapses[old_synapse_index]
        old_synapse.disable()

        start_neuron = old_synapse.get_start()
        end_neuron = old_synapse.get_end()

        new_neuron = HiddenNeuron()

        new_weight_1 = 1
        new_weight_2 = old_synapse.get_weight()

        if old_is_input:
            new_index = 0
            self.__hidden_neurons.insert(0, new_neuron)
            for synapse_index, index_1, is_input, index_2, is_output in enumerate(self.__symbolic_synapses):
                if not is_input:
                    index_1 += 1
                if not is_output:
                    index_2 += 1
                self.__symbolic_synapses[synapse_index] = (index_1, is_input, index_2, is_output)
        elif old_is_output:
            new_index = len(self.__hidden_neurons)
            self.__hidden_neurons.append(new_neuron)
        else:
            new_index = int((old_index_1+old_index_2)/2.0 + 1)
            self.__hidden_neurons. insert(new_index, new_neuron)
            for synapse_index, index_1, is_input, index_2, is_output in enumerate(self.__symbolic_synapses):
                if index_1 >= new_index and not is_input:
                    index_1 += 1
                if index_2 >= new_index and not is_output:
                    index_2 += 1
                self.__symbolic_synapses[synapse_index] = (index_1, is_input, index_2, is_output)

        self.__synapses.append(GeneticSynapse(start_neuron, new_weight_1, new_neuron))
        self.__symbolic_synapses.append((old_index_1, old_is_input, new_index, False))
        self.__synapses.append(GeneticSynapse(new_neuron, new_weight_2, end_neuron))
        self.__symbolic_synapses.append((new_index, False, old_index_2, old_is_output))

    def crossover(self, other: "Network") -> None:
        if other.get_io_signature() != self.get_io_signature():
            raise ValueError("The networks have different input/output signatures")

        if self.get_fitness() < other.get_fitness():
            raise ValueError("The network to cross over has lower fitness than it's parent")

        min_innovation = min(self.get_innovation(), other.get_innovation())
        for i in range(min_innovation):
            self_symbolic_synapse = self.__symbolic_synapses[i]
            other_symbolic_synapse = other.__symbolic_synapses[i]

            if self_symbolic_synapse == other_symbolic_synapse and \
                    self.__synapses[i].is_available() != other.__synapses[i].is_available():
                if self.__generator.randint(1, 100) <= 75:
                    self.__synapses[i].disable()
                else:
                    self.__synapses[i].enable()

            if self.get_fitness() == other.get_fitness() and self_symbolic_synapse != other_symbolic_synapse:
                if self.__generator.randint(1, 100) <= 50:
                    index_1, is_input, index_2, is_output = other_symbolic_synapse
                    start_neuron = self.__input_neurons[index_1] if is_input else self.__hidden_neurons[index_1]
                    weight = other.__synapses[i].get_weight()
                    end_neuron = self.__output_neurons[index_2] if is_output else self.__hidden_neurons[index_2]
                    self.__synapses[i] = GeneticSynapse(start_neuron, weight, end_neuron)
                    self.__symbolic_synapses[i] = (index_1, is_input, index_2, is_output)

        if self.get_fitness() == other.get_fitness():
            if self.get_innovation() == min_innovation:
                for i in range(other.get_innovation()-min_innovation):
                    # if self.__generator.randint(1, 100) <= 50:
                    i += min_innovation
                    index_1, is_input, index_2, is_output = other.__symbolic_synapses[i]
                    if not is_input and index_1 >= self.get_innovation():
                        index_1 = self.get_innovation()
                        self.__hidden_neurons.append(HiddenNeuron())
                        if not is_output and index_2 == index_1:
                            index_2 += 1
                    if not is_output and index_2 >= self.get_innovation():
                        index_2 = self.get_innovation()
                        self.__hidden_neurons.append(HiddenNeuron())
                    start_neuron = self.__input_neurons[index_1] if is_input else self.__hidden_neurons[index_1]
                    weight = other.__synapses[i].get_weight()
                    end_neuron = self.__output_neurons[index_2] if is_output else self.__hidden_neurons[index_2]
                    self.__synapses[i] = GeneticSynapse(start_neuron, weight, end_neuron)
                    self.__symbolic_synapses[i] = (index_1, is_input, index_2, is_output)
            # else:
            #     for i in range(self.get_innovation()-min_innovation):
            #         if self.__generator.randint(1, 100) <= 50:
            #             i += min_innovation
            #             index_1, is_input, index_2, is_output = self.__symbolic_synapses[i]
            #             index_1_found = False
            #             index_2_found = False
            #             for j, i1, _, i2, _ in enumerate(self.__symbolic_synapses):
            #                 if index_1 == i1 and j != i:
            #                     index_1_found = True
            #                 if index_2 == i2 and j != i:
            #                     index_2_found = True
            #             if not (is_input or index_1_found) or not (is_output or index_2_found):
            #                 not_found = index_1 if index_2_found else index_2
            #                 self.__hidden_neurons.pop(not_found)
            #                 for synapse_index, index_1_2, is_input_2, index_2_2, is_output_2 in enumerate(
            #                         self.__symbolic_synapses):
            #                     if index_1_2 >= not_found and not is_input_2:
            #                         index_1_2 -= 1
            #                     if index_2_2 >= not_found and not is_output_2:
            #                         index_2_2 -= 1
            #                     self.__symbolic_synapses[synapse_index] = (index_1_2, is_input_2,
            #                                                                index_2_2, is_output_2)
            #             self.__synapses.pop(i)
            #             self.__symbolic_synapses.pop(i)

    @staticmethod
    def compare(n1: "Network", n2: "Network") -> Tuple[float, float, float]:
        if n1.get_io_signature() != n2.get_io_signature():
            raise ValueError("The networks being compared have different input/output signatures")
        min_innovation = min(n1.get_innovation(), n2.get_innovation())
        max_innovation = max(n1.get_innovation(), n2.get_innovation())
        excess = max_innovation-min_innovation
        disjoint = 0
        weight_1 = 0
        weight_2 = 0
        for i in range(min_innovation):
            weight_1 += n1.__synapses[i].get_weight()
            weight_2 += n2.__synapses[i].get_weight()
            if n1.__symbolic_synapses[i] != n2.__symbolic_synapses[i]:
                disjoint += 1
        if n1.get_innovation() == max_innovation:
            for i in range(max_innovation-min_innovation):
                i += min_innovation
                weight_1 += n1.__synapses[i].get_weight()
        else:
            for i in range(max_innovation-min_innovation):
                i += min_innovation
                weight_2 += n2.__synapses[i].get_weight()
        weight_1 /= n1.get_innovation()
        weight_2 /= n2.get_innovation()
        average_weight_difference = abs(weight_1-weight_2)
        return excess/max_innovation, disjoint/max_innovation, average_weight_difference

    def set_seed(self, seed: int) -> None:
        self.__generator.seed(seed)

    def set_fitness(self, fitness: float) -> None:
        self.__fitness = fitness

    def set_shared_fitness(self, shared_fitness: float) -> None:
        self.__shared_fitness = shared_fitness

    def get_random_float(self) -> float:
        return self.__generator.uniform(-2.0, 2.0)

    def get_seed(self) -> int:
        return self.__generator.getstate()

    def get_fitness(self) -> float:
        return self.__fitness

    def get_shared_fitness(self) -> float:
        return self.__shared_fitness

    def get_birth_generation(self) -> int:
        return self.__birth_generation

    def get_io_signature(self) -> Tuple[int, int]:
        return len(self.__input_neurons), len(self.__output_neurons)

    def get_innovation(self) -> int:
        return len(self.__synapses)
