import random
from typing import List, Any, Tuple, Union


def find(l: List[Any], obj: Any) -> int:
    """
    Function for finding a value inside a list.

    :param l: List to look through
    :param obj: Value to find
    :return: The value's index if found, -1 if not
    """
    try:
        return l.index(obj)
    except ValueError:
        return -1


class GeneticNeuron:

    __calculations: int

    # Constructor

    def __init__(self):
        """
        Constructor of a Genetic Neuron.
        """
        self.__calculations = 0

    # Calculations counter

    def set_calculations(self, calculations: int) -> None:
        """
        Sets the amount of calculations performed by the Neuron.

        :param calculations: New amount
        """
        self.__calculations = calculations

    def get_calculations(self) -> int:
        """
        Gets the amount of calculations performed by the Neuron.

        :return: Amount of calculations
        """
        return self.__calculations

    # Main behaviour

    def calculate(self) -> None:
        """
        Calculates once.
        """
        self.__calculations += 1

    @staticmethod
    def modified_sigmoid(x: float) -> float:
        """
        Static mathematical function of use when normalizing values.

        :param x: Input
        :return: Output
        """
        import numpy as np
        import math
        return math.exp(-np.logaddexp(0.0, -4.9*x))

    def add_input(self, synapse: "GeneticSynapse") -> None:
        """
        Adds an input to the inputs list in subclasses.

        :param synapse: New input synapse
        """
        pass

    def add_output(self, synapse: "GeneticSynapse") -> None:
        """
        Adds an output to the outputs list in subclasses.

        :param synapse: New output synapse
        """
        pass

    def remove_input(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an input from the inputs list in subclasses.

        :param synapse: Old input synapse
        """
        pass

    def remove_output(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an output from the outputs list in subclasses.

        :param synapse: Old output synapse
        """
        pass


class InputNeuron(GeneticNeuron):

    __outputs: List["GeneticSynapse"]

    __value: float

    def __init__(self):
        """
        Constructor for an Input Neuron.

        Sets default values.
        """
        super().__init__()
        self.__outputs = []
        self.__value = 0.0

    def set_value(self, value: float) -> None:
        """
        Sets a value for the Neuron to spread.

        :param value: New input value of the network.
        """
        self.__value = value

    def add_output(self, synapse: "GeneticSynapse") -> None:
        """
        Adds an output to relay to.

        :param synapse: New output synapse
        """
        self.__outputs.append(synapse)

    def remove_output(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an output from the outputs list.

        :param synapse: Old output synapse
        """
        self.__outputs.remove(synapse)

    def calculate(self):
        """
        Simply relays the value to all outputs.
        """
        super().calculate()
        for o in self.__outputs:
            o.relay(self.__value)


class HiddenNeuron(GeneticNeuron):

    __outputs: List["GeneticSynapse"]
    __inputs: List["GeneticSynapse"]

    def __init__(self):
        """
        Constructor for a Hidden Neuron, sets default values
        """
        super().__init__()
        self.__inputs = []
        self.__outputs = []

    def add_input(self, synapse: "GeneticSynapse") -> None:
        """
        Adds a new input to relay values from.

        :param synapse: New input synapse
        """
        self.__inputs.append(synapse)

    def add_output(self, synapse: "GeneticSynapse") -> None:
        """
        Adds a new output to relay values to.

        :param synapse: New output synapse
        """
        self.__outputs.append(synapse)

    def remove_input(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an input from the inputs list.

        :param synapse: Old input synapse
        """
        self.__inputs.remove(synapse)

    def remove_output(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an output from the outputs list.

        :param synapse: Old output synapse
        """
        self.__outputs.remove(synapse)

    def calculate(self) -> None:
        """
        Relays values from inputs, adds them up, normalizes the output, and then relays it out.
        """
        super().calculate()

        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.fetch()

        result = self.modified_sigmoid(weighted_sum)
        for o in self.__outputs:
            o.relay(result)


class OutputNeuron(GeneticNeuron):

    __inputs: List["GeneticSynapse"]

    __result: bool

    def __init__(self):
        """
        Constructor for an Output Neuron, for the final layer of the network.
        """
        super().__init__()
        self.__inputs = []
        self.__result = False

    def add_input(self, synapse: "GeneticSynapse") -> None:
        """
        Adds a new input to fetch values from.

        :param synapse: New input synapse
        """
        self.__inputs.append(synapse)

    def remove_input(self, synapse: "GeneticSynapse") -> None:
        """
        Removes an input from the inputs list.

        :param synapse: Old input synapse
        """
        self.__inputs.remove(synapse)

    def calculate(self) -> None:
        """
        Relays values from inputs, adds them up, normalizes the output, turns it to bool, and saves it.
        """
        super().calculate()

        weighted_sum = 0
        for i in self.__inputs:
            weighted_sum += i.fetch()

        self.__result = self.modified_sigmoid(weighted_sum) > 0.5

    def get_result(self) -> bool:
        """
        Gets the last result calculated.

        :return: Last result calculated by the output neuron
        """
        return self.__result


class GeneticSynapse:

    __availability: bool
    __weighted_value: float
    __end_neuron: GeneticNeuron
    __weight: float
    __start_neuron: GeneticNeuron

    # Constructor

    def __init__(self, start_neuron: GeneticNeuron, weight: float, end_neuron: GeneticNeuron):
        """
        Constructor for a synapse.

        Sets default values.

        :param start_neuron: Input neuron
        :param weight: Weight for calculations
        :param end_neuron: Output neuron
        """
        self.__start_neuron = start_neuron
        self.__weight = weight
        self.__end_neuron = end_neuron

        # Makes sure neurons have access to the synapse
        start_neuron.add_output(self)
        end_neuron.add_input(self)

        self.__weighted_value = 0.0
        self.__availability = True

    # Relay system

    def relay(self, result: float) -> None:
        """
        Sets a new weighted value in the synapse.

        :param result: New value to weigh
        """
        self.__weighted_value = self.__weight * result

    def fetch(self) -> float:
        """
        Fetches a weighted value from the input.

        Forces cascade-like recursive calculations using the 'calculations' parameter.

        :return: The latest weighed value if enabled, else 0
        """
        if self.__start_neuron.get_calculations() < self.__end_neuron.get_calculations():
            self.__start_neuron.calculate()
        return self.__weighted_value if self.is_available() else 0.0

    # Availability management

    def enable(self) -> None:
        """
        Turns a synapse on, allowing it to relay values.
        """
        self.__availability = True

    def disable(self) -> None:
        """
        Turns a synapse off, shutting down relays.
        """
        self.__availability = False

    def is_available(self) -> bool:
        """
        Checks if the synapse is enabled.

        :return: True if available, False if not
        """
        return self.__availability

    # SETTERS

    def set_start(self, start_neuron: GeneticNeuron) -> None:
        """
        Sets the input neuron of the synapse.

        :param start_neuron: New input neuron
        """
        self.__start_neuron = start_neuron

    def set_weight(self, weight: float) -> None:
        """
        Sets the synapse's weight value.

        :param weight: New synapse weight.
        """
        self.__weight = weight

    def set_end(self, end_neuron: GeneticNeuron) -> None:
        """
        Sets the synapses output neuron.

        :param end_neuron: New output neuron.
        """
        self.__end_neuron = end_neuron

    # GETTERS

    def get_start(self) -> GeneticNeuron:
        """
        Gets the input neuron of the synapse.

        :return: Input neuron of the synapse
        """
        return self.__start_neuron

    def get_weight(self) -> float:
        """
        Gets the synapse's weight.

        :return: Weight of the synapse
        """
        return self.__weight

    def get_end(self) -> GeneticNeuron:
        """
        Gets the output neuron of the synapse.

        :return: Output neuron of the synapse
        """
        return self.__end_neuron


class Network:

    # All Neurons in the network
    __input_neurons: List[InputNeuron] = []
    __hidden_neurons: List[HiddenNeuron] = []
    __output_neurons: List[OutputNeuron] = []
    __calculations: int

    # All synapses in the network
    __synapses: List[GeneticSynapse] = []

    # Abstract representation of a synapse, using indexes of neurons in the network's lists as ids
    __symbolic_synapses: List[Tuple[int, bool, int, bool]] = []

    # Pseudo-random number generator
    __generator: random.Random = random.Random()

    # Fitness of the network as a classifier
    __fitness: float = 0.0

    # Shared fitness of the network, altered by it's species
    __shared_fitness: float = 0.0

    # CONSTRUCTOR

    def __init__(self, input_amount: int, output_amount: int, seed: int = None):
        """
        Constructor of a Network, sets up default parameters.

        :param input_amount: Amount of inputs to receive
        :param output_amount: Amount of outputs to produce
        :param seed: Seed of the pseudo-random generator
        """
        # All Neurons in the network
        self.__input_neurons = []
        self.__hidden_neurons = []
        self.__output_neurons = []
        self.__calculations = 0

        # All synapses in the network
        self.__synapses = []

        # Abstract representation of a synapse, using indexes of neurons in the network's lists as ids
        self.__symbolic_synapses = []

        # Pseudo-random number generator
        self.__generator = random.Random()

        # Fitness of the network as a classifier
        self.__fitness = 0.0

        # Shared fitness of the network, altered by it's species
        self.__shared_fitness = 0.0

        # Seed setting
        if seed is not None:
            self.set_seed(seed)

        # Initial input neurons
        for _ in range(input_amount):
            self.__input_neurons.append(InputNeuron())

        # Initial output neurons
        for _ in range(output_amount):
            self.__output_neurons.append(OutputNeuron())

    def __initialize_synapses(self):
        """
        Sets up new, randomized, initial synapses for the network.
        """
        # Initial synapses from input to output
        for i_index, i in enumerate(self.__input_neurons):
            for o_index, o in enumerate(self.__output_neurons):
                self.__synapses.append(GeneticSynapse(i, self.get_random_float(), o))
                self.__symbolic_synapses.append((i_index, True, o_index, True))

    # FACTORY METHOD

    @staticmethod
    def new(input_amount: int, output_amount: int, seed: int = None) -> "Network":
        """
        Generates a new, functional network.

        Should be the first method of construction used.

        :param input_amount: Length of an acceptable input list
        :param output_amount: Length of an acceptable output list
        :param seed: Pseudo-random generator seed
        :return: A new network
        """
        n = Network(input_amount, output_amount, seed)
        n.__initialize_synapses()
        return n

    # CLONING

    def clone(self) -> "Network":
        """
        Clones the network, copying all values including reduced base fitnesses.

        :return: A new network, identical to self
        """
        # Constructs a copy of the network, weights in synapses are the same too
        new = Network(len(self.__input_neurons), len(self.__output_neurons), seed=self.get_seed())

        # Add the same amount of hidden neurons
        for _ in self.__hidden_neurons:
            new.__hidden_neurons.append(HiddenNeuron())

        # For every synapse in self, an homologous is constructed with the same properties
        for synapse_index, (index_1, is_input, index_2, is_output) in enumerate(self.__symbolic_synapses):
            # It's I/O scheme is found
            if is_input:
                start_neuron = new.__input_neurons[index_1]
            else:
                start_neuron = new.__hidden_neurons[index_1]
            if is_output:
                end_neuron = new.__output_neurons[index_2]
            else:
                end_neuron = new.__hidden_neurons[index_2]

            # The synapse is added
            new.__synapses.append(GeneticSynapse(start_neuron, self.__synapses[synapse_index].get_weight(), end_neuron))

            # Availability is respected
            if not self.__synapses[synapse_index].is_available():
                new.__synapses[synapse_index].disable()

            # The symbolic counterpart is added too
            new.__symbolic_synapses.append((index_1, is_input, index_2, is_output))

        new.__fitness = 0.95*self.__fitness
        new.__shared_fitness = 0.95*self.__shared_fitness

        return new

    # MAIN EXECUTION

    def calculate(self, inputs: List[Union[int, float]]) -> List[bool]:
        """
        Calculates output of the network for a given input.

        :param inputs: List of floating point inputs, assumed to be normalized
        :return: List of bool outputs
        """
        # Check if the contracts match
        if len(inputs) != len(self.__input_neurons):
            raise RuntimeError("Wrong input length, doesn't correspond with the network's signature")

        self.__calculations += 1

        # The input values are set
        for index, i in enumerate(self.__input_neurons):
            i.set_value(inputs[index])

        # The output is calculated recursively
        results = []
        for o in self.__output_neurons:
            o.calculate()
            results.append(o.get_result())

        return results

    # MUTATIONS

    def mutation_change_weights(self) -> None:
        """
        Mutates the network, changing random weights within it.
        """
        for synapse in self.__synapses:
            # Chance of linear perturbation
            if self.choose_with_probability(99.0):
                perturbation = 0.01 if self.choose_with_probability(50.0) else -0.01
                new_weight = synapse.get_weight() + perturbation
            # Chance of re-roll
            else:
                new_weight = self.get_random_float()
            # The weights are set
            synapse.set_weight(new_weight)

    def mutation_add_synapse(self) -> None:
        """
        Adds a new synapse at a random available position, if the position was taken, the weight of the original
        synapse is re-rolled, and the synapse is re-activated too.
        """
        # If there's enough hidden neurons to choose among them
        if len(self.__hidden_neurons) > 1:
            i = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__input_neurons) - 1)
            o = self.__generator.randint(0, len(self.__hidden_neurons) + len(self.__output_neurons) - 1)

            is_input = not i < len(self.__hidden_neurons)
            is_output = not o < len(self.__hidden_neurons)

            if is_input:
                i -= len(self.__hidden_neurons)
            if is_output:
                o -= len(self.__hidden_neurons)
        # If only I/O neurons are available
        else:
            i = self.__generator.randint(0, len(self.__input_neurons) - 1)
            o = self.__generator.randint(0, len(self.__output_neurons) - 1)

            is_input = True
            is_output = True

        # If both are hidden, they need to be different, and synapses always go in one direction of the indices only
        if not (is_input or is_output):
            while i >= o:
                i = self.__generator.randint(0, len(self.__hidden_neurons) - 1)
                o = self.__generator.randint(0, len(self.__hidden_neurons) - 1)

        # For comparison among other symbolic synapses
        symbolic_representation = (i, is_input, o, is_output)

        # Searches for already used synapses
        for synapse_index, s in enumerate(self.__symbolic_synapses):
            # If one is found, nothing needs to be added
            if s == symbolic_representation:
                self.__synapses[synapse_index].enable()
                self.__synapses[synapse_index].set_weight(self.get_random_float())
                return

        # Neurons are found
        if is_input:
            start_neuron = self.__input_neurons[i]
        else:
            start_neuron = self.__hidden_neurons[i]
        if is_output:
            end_neuron = self.__output_neurons[o]
        else:
            end_neuron = self.__hidden_neurons[o]

        # The new synapse is added
        weight = self.get_random_float()
        self.__synapses.append(GeneticSynapse(start_neuron, weight, end_neuron))
        self.__symbolic_synapses.append(symbolic_representation)

    def mutation_add_neuron(self) -> None:
        """
        Adds a new neuron at a random available position inside a synapse.

        If the synapse is disabled it stays that way.
        """
        # Chooses a random synapse
        old_synapse_index = self.__generator.randint(0, len(self.__synapses) - 1)
        old_synapse = self.__synapses[old_synapse_index]

        # Finds it's symbolic representation
        old_index_1, old_is_input, old_index_2, old_is_output = self.__symbolic_synapses[old_synapse_index]

        # Disables the old connection
        old_synapse.disable()

        # The old synapse's values are found
        start_neuron = old_synapse.get_start()
        end_neuron = old_synapse.get_end()

        # The new neuron is created
        new_neuron = HiddenNeuron()
        new_neuron.set_calculations(self.get_calculations())

        # Weights are generated
        new_weight_1 = 1.0
        new_weight_2 = old_synapse.get_weight()

        # If the old synapse started at an input, the new neuron is appended to the list at it's beginning
        if old_is_input:
            new_index = 0
            self.__hidden_neurons.insert(0, new_neuron)
            # All other neuron indexes must be moved by 1 up
            for synapse_index, (index_1, is_input, index_2, is_output) in enumerate(self.__symbolic_synapses):
                if not is_input:
                    index_1 += 1
                if not is_output:
                    index_2 += 1
                # Symbolic synapses are replaced
                self.__symbolic_synapses[synapse_index] = (index_1, is_input, index_2, is_output)
        # If the old synapse ended in an output, the new neuron is appended at the end of the list
        elif old_is_output:
            new_index = len(self.__hidden_neurons)
            self.__hidden_neurons.append(new_neuron)
        # In any other case, the new neuron is appended in the middle of the list
        else:
            new_index = int((old_index_1+old_index_2)/2.0 + 1)
            self.__hidden_neurons. insert(new_index, new_neuron)
            # All neuron indexes that were equal or higher to the new neuron's need to be pushed up by 1
            for synapse_index, (index_1, is_input, index_2, is_output) in enumerate(self.__symbolic_synapses):
                if index_1 >= new_index and not is_input:
                    index_1 += 1
                if index_2 >= new_index and not is_output:
                    index_2 += 1
                # Symbolic synapses are replaced
                self.__symbolic_synapses[synapse_index] = (index_1, is_input, index_2, is_output)

        # The new synapses are created, enabled by default
        self.__synapses.append(GeneticSynapse(start_neuron, new_weight_1, new_neuron))
        self.__symbolic_synapses.append((old_index_1, old_is_input, new_index, False))
        self.__synapses.append(GeneticSynapse(new_neuron, new_weight_2, end_neuron))
        self.__symbolic_synapses.append((new_index, False, old_index_2, old_is_output))

    # CROSSOVER

    def crossover(self, other: "Network", share_disjoins: bool) -> None:
        """
        Adds synapses from 'other' into self's synapse list.

        :param other: Another network from which to take synapses and neuron positions
        """
        # Check if the contracts match
        if other.get_io_signature() != self.get_io_signature():
            raise ValueError("The networks have different input/output signatures")

        # Smallest amount of synapses among 'self' and 'other'
        min_innovation = min(self.get_innovation(), other.get_innovation())

        # For all synapses that are within the common index range
        for i in range(min_innovation):
            # Symbolic synapses are found
            self_symbolic_synapse = self.__symbolic_synapses[i]
            other_symbolic_synapse = other.__symbolic_synapses[i]

            # If they are the same, and only one of them is available, self's one gets a 50% chance of being disabled
            if self_symbolic_synapse == other_symbolic_synapse and \
                    self.__synapses[i].is_available() != other.__synapses[i].is_available():
                if self.choose_with_probability(50.0):
                    self.__synapses[i].disable()
                else:
                    self.__synapses[i].enable()

            # If they are not the same, and the disjoints will be shared
            if share_disjoins and self_symbolic_synapse != other_symbolic_synapse:
                # There's a 50% chance of swapping for the other one
                if self.choose_with_probability(50.0):

                    # Other's symbolic synapse is disassembled
                    index_1, is_input, index_2, is_output = other_symbolic_synapse

                    # If the neurons involved don't match up with what's available in 'self', the transaction is skipped
                    if (not is_input and index_1 >= len(self.__hidden_neurons)) or \
                            (not is_output and index_2 >= len(self.__hidden_neurons)):
                        continue

                    # If they match, a copy of other's synapse is created
                    start_neuron = self.__input_neurons[index_1] if is_input else self.__hidden_neurons[index_1]
                    weight = other.__synapses[i].get_weight()
                    end_neuron = self.__output_neurons[index_2] if is_output else self.__hidden_neurons[index_2]

                    # The old synapse is disconnected from the neurons
                    self.__synapses[i].get_start().remove_output(self.__synapses[i])
                    self.__synapses[i].get_end().remove_input(self.__synapses[i])

                    # It's replaced, maintaining availability
                    self.__synapses[i] = GeneticSynapse(start_neuron, weight, end_neuron)
                    if not other.__synapses[i].is_available():
                        self.__synapses[i].disable()
                    self.__symbolic_synapses[i] = (index_1, is_input, index_2, is_output)

    # COMPARATOR

    @staticmethod
    def compare(network_1: "Network", network_2: "Network") -> Tuple[float, float, float]:
        """
        Compares two networks based on the amount of common synapses they have, their innovation numbers,
        and their average weight differences.

        :param network_1: First network to compare
        :param network_2: Second network to compare
        :return: Tuple of three metrics of distance for using to separate species
        """
        # Checks if the signatures don't match
        if network_1.get_io_signature() != network_2.get_io_signature():
            raise ValueError("The networks being compared have different input/output signatures")

        # Finds innovation differences
        min_innovation = min(network_1.get_innovation(), network_2.get_innovation())
        max_innovation = max(network_1.get_innovation(), network_2.get_innovation())

        # The first metric is calculated
        excess_distance = (max_innovation-min_innovation)/float(max_innovation)

        # Disjoint synapses are counted, and average weight difference is taken into account too
        disjoint = 0
        weight_1 = 0
        weight_2 = 0

        # The common range synapses are checked
        for i in range(min_innovation):
            weight_1 += network_1.__synapses[i].get_weight()
            weight_2 += network_2.__synapses[i].get_weight()
            if network_1.__symbolic_synapses[i] != network_2.__symbolic_synapses[i]:
                disjoint += 1

        # The rest are checked
        if network_1.get_innovation() == max_innovation:
            for i in range(max_innovation-min_innovation):
                i += min_innovation
                weight_1 += network_1.__synapses[i].get_weight()
        else:
            for i in range(max_innovation-min_innovation):
                i += min_innovation
                weight_2 += network_2.__synapses[i].get_weight()

        # Values are normalized
        disjoint_distance = disjoint/float(max_innovation)
        weight_1 /= float(network_1.get_innovation())
        weight_2 /= float(network_2.get_innovation())
        average_weight_difference = abs(weight_1-weight_2)

        return excess_distance, disjoint_distance, average_weight_difference

    # SETTERS

    def set_seed(self, seed: int) -> None:
        """
        Sets a network's pseudo-random seed.

        :param seed: New seed
        """
        self.__generator.seed(seed)

    def set_fitness(self, fitness: float) -> None:
        """
        Sets the fitness of the classifier.

        :param fitness: New fitness
        """
        self.__fitness = fitness

    def set_shared_fitness(self, shared_fitness: float) -> None:
        """
        Sets the shared fitness of the classifier.

        :param shared_fitness: New shared fitness
        """
        self.__shared_fitness = shared_fitness

    # GETTERS

    def get_seed(self) -> int:
        """
        Gets the current pseudo-random seed of the network.

        :return: The seed
        """
        return self.__generator.getstate()[0]

    def get_fitness(self) -> float:
        """
        Gets the fitness of the network classifier.

        :return: The network's fitness
        """
        return self.__fitness

    def get_shared_fitness(self) -> float:
        """
        Gets the shared fitness of the network.

        :return: The network's shared fitness
        """
        return self.__shared_fitness

    def get_io_signature(self) -> Tuple[int, int]:
        """
        Gets the I/O data of the network, for comparing compatible networks.

        :return: Tuple of input length and output length
        """
        return len(self.__input_neurons), len(self.__output_neurons)

    def get_innovation(self) -> int:
        """
        Gets the current innovation metric of the network.

        :return: The network's innovation
        """
        return len(self.__synapses)

    def get_calculations(self) -> int:
        """
        Gets the amount of calculations performed by the network.

        :return: The amount of calculations to date
        """
        return self.__calculations

    # PROBABILITY MANAGERS

    def choose_with_probability(self, probability_percentage: float) -> bool:
        """
        Returns True only with a certain probability percentage given.

        :param probability_percentage: Percentage probability of returning True
        :return: True or False, depending on probability
        """
        return self.__generator.uniform(0, 100) < probability_percentage

    def get_random_float(self) -> float:
        """
        Return a random uniformly distributed float number between -2 and 2.

        :return: A random float in [-2, 2]
        """
        return self.__generator.uniform(-2.0, 2.0)
