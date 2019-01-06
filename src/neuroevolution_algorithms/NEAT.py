import random
from random import Random
from typing import List, Optional, Callable, Union, Tuple

from neuroevolution_algorithms.neat_network import Network


class Neat:
    """
    Class storing the N.E.A.T. algorithm.
    """

    # Input/output length
    __input_amount: int
    __output_amount: int

    # Fitness evaluation function
    __fitness_function: Callable[[List[Network]], List[Union[float, int]]]

    # Population configuration
    __population_size: int
    __fitness_stagnancy_cap: int
    __large_species_size: int

    # Mutation probability percentages
    __mutation_change_weights_chance: float
    __mutation_add_synapse_chance: float
    __mutation_add_neuron_chance: float

    # Reproduction probability percentages
    __no_crossover_chance: float
    __inter_species_mating_chance: float

    # Species separation distance
    __excess_distance_coefficient: float
    __disjoint_distance_coefficient: float
    __weights_distance_coefficient: float

    __species_distance_cap: float

    # Pseudo-random generation
    __starting_seed: Optional[int]
    __seed: Optional[int]
    __generator: Random

    # Evolution stage
    __generation: int

    # Stored population
    __population: List[Network]
    __symbolic_species: List[List[int]]

    # Stored fitnesses
    __shared_fitness_sums: List[float]
    __total_shared_fitness: float

    # CONSTRUCTOR

    def __init__(self, input_amount: int, output_amount: int,
                 fitness_function: Callable[[List[Network]], List[Union[float, int]]],
                 population_size: int = 150, fitness_stagnancy_cap: int = 15, large_species_size: int = 5,
                 mutation_change_weights_chance: float = 80.0, mutation_add_synapse_chance: float = 5.0,
                 mutation_add_neuron_chance: float = 3.0, no_crossover_chance: float = 25.0,
                 inter_species_mating_chance: float = 0.1, excess_distance_coefficient: float = 1.0,
                 disjoint_distance_coefficient: float = 1.0, weights_distance_coefficient: float = 0.4,
                 species_distance_cap: float = 3.0, seed: int = None):
        """
        Constructor method for a NEAT algorithm.

        It's better to use the '.builder()' method for generating NEATs.

        :param input_amount: Amount of inputs for the networks
        :param output_amount: Amount of outputs ofr the networks
        :param fitness_function: Fitness function for networks
        :param population_size: Size of the overall population
        :param fitness_stagnancy_cap: Cap of stagnancy as a relative fitness percentage
        :param large_species_size: Size from which a species can be considered large
        :param mutation_change_weights_chance: Chance percentage of the weights of a newborn network of mutating
        :param mutation_add_synapse_chance: Chance percentage of a newborn network to have a new synapse mutated
        :param mutation_add_neuron_chance: Chance percentage of a newborn network to have a new neuron mutated
        :param no_crossover_chance: Chance percentage of a newborn network to have no crossover between parents
        :param inter_species_mating_chance: Chance percentage of a newborn network to be bork of inter-species parents
        :param excess_distance_coefficient: Coefficient magnifying the effect of excess distance in species separation
        :param disjoint_distance_coefficient: Coefficient magnifying the effect of disjoint distance in \
        species separation
        :param weights_distance_coefficient: Coefficient magnifying the effect of weight distance in species separation
        :param species_distance_cap: Distance cap between members of different species
        :param seed: Seed for pseudo-random number generation
        """
        # The values are saved
        self.__input_amount = input_amount
        self.__output_amount = output_amount

        self.__fitness_function = fitness_function

        self.__population_size = population_size
        self.__fitness_stagnancy_cap = fitness_stagnancy_cap
        self.__large_species_size = large_species_size

        self.__mutation_change_weights_chance = mutation_change_weights_chance
        self.__mutation_add_synapse_chance = mutation_add_synapse_chance
        self.__mutation_add_neuron_chance = mutation_add_neuron_chance

        self.__no_crossover_chance = no_crossover_chance
        self.__inter_species_mating_chance = inter_species_mating_chance

        self.__excess_distance_coefficient = excess_distance_coefficient
        self.__disjoint_distance_coefficient = disjoint_distance_coefficient
        self.__weights_distance_coefficient = weights_distance_coefficient

        self.__species_distance_cap = species_distance_cap

        self.__generator = random.Random()

        if seed is not None:
            self.__seed = seed
        else:
            self.__seed = self.__generator.randint(0, 100000)

        self.__starting_seed = self.__seed
        self.__generator.seed(self.__seed)

        self.__generation = 0

        self.__population = []
        self.__symbolic_species = []

        self.__shared_fitness_sums = [0.0]
        self.__total_shared_fitness = 0.0

        # The population is initialized
        first_species = []
        for i in range(population_size):
            n = Network.new(input_amount, output_amount, seed=self.__seed)
            self.__seed += 1
            self.__population.append(n)
            first_species.append(i)
        self.__symbolic_species.append(first_species)

    # MAIN BEHAVIOUR

    def advance_generation(self) -> None:
        """
        Advances the population by a single generation.

        First it cleans useless species, then reproduces those who are left, re-calculates their fitnesses, sorts
        them, separates species, and finally calculates the shared fitnesses of all networks.

        Must be run once before any evaluation of metrics.
        """
        if self.__generation == 0:
            self.__calculate_fitnesses()
            self.__sort_population_by_fitness()
            self.__separate_species()
            self.__calculate_shared_fitnesses()
        else:
            self.__remove_useless_species()
            self.__reproduce_population()
            self.__calculate_fitnesses()
            self.__sort_population_by_fitness()
            self.__separate_species()
            self.__calculate_shared_fitnesses()
        self.__generation += 1

    # STEPS OF GENERATION ADVANCEMENT

    def __remove_useless_species(self) -> None:
        """
        Sets a species as removable by changing it's shared fitness sum to 0.

        Removes large species with under-achieving fitnesses from the population
        """
        for index, s in enumerate(self.__symbolic_species):
            best = max(s)
            if best <= 0.75*self.__population_size and len(self.__shared_fitness_sums) >= self.__large_species_size:
                self.__total_shared_fitness -= self.__shared_fitness_sums[index]
                self.__shared_fitness_sums[index] = 0.0
                for n_index in s:
                    self.__population[n_index].set_shared_fitness(0.0)

    def __reproduce_population(self) -> None:
        """
        Generates a new population through reproducing members from the last generation.
        """
        new_population = []

        # Checks every species
        for index, s in enumerate(self.__symbolic_species):

            # If not all networks performed bad or if this species is allowed to reproduce
            if self.__shared_fitness_sums[index] != 0.0 or self.__total_shared_fitness == 0.0:

                # The best element may be kept if the species is large
                keep_best = len(s) >= self.__large_species_size

                # Amount of children
                offspring_amount = int(self.__population_size * (self.__shared_fitness_sums[index] /
                                                                 self.__total_shared_fitness)
                                       if self.__total_shared_fitness != 0
                                       else self.__population_size * (1.0 / len(self.__symbolic_species)))

                # Some space is left for later introducing variability
                offspring_amount -= int(self.__population_size*0.01)

                # The best may be kept
                if keep_best:
                    best_i = max(s)
                    new_population.append(self.__population[best_i])
                    offspring_amount -= 1

                # Children are added
                for _ in range(offspring_amount):

                    # Chance of no crossover
                    if len(s) == 1 or self.choose_with_probability(self.__no_crossover_chance):
                        child = self.__population[s[self.__generator.randint(0, len(s) - 1)]].clone()
                    # If crossover
                    else:
                        # A main parent os chosen
                        child_index = s[self.__generator.randint(0, len(s) - 1)]
                        child = self.__population[child_index]

                        # Chance of inter-species reproduction
                        if self.choose_with_probability(self.__inter_species_mating_chance):
                            mate_index = self.__generator.randint(0, len(self.__population) - 1)

                            # Only authorized networks are accepted
                            while mate_index == child_index or (
                                    self.__population[mate_index].get_shared_fitness() == 0.0 and
                                    not self.__total_shared_fitness == 0.0):
                                mate_index = self.__generator.randint(0, len(self.__population) - 1)
                        # If normal reproduction
                        else:
                            # A secondary parent is found
                            mate_index = s[self.__generator.randint(0, len(s) - 1)]
                            while mate_index == child_index:
                                mate_index = s[self.__generator.randint(0, len(s) - 1)]
                        mate = self.__population[mate_index]

                        # They may have equal importance
                        share_disjoints = child.get_fitness() == mate.get_fitness()

                        # Reproduction
                        if child.get_fitness() >= mate.get_fitness():
                            child = child.clone()
                            child.crossover(mate, share_disjoints)
                        else:
                            mate = mate.clone()
                            mate.crossover(child, share_disjoints)
                            child = mate

                    # Mutations may happen
                    if self.choose_with_probability(self.__mutation_change_weights_chance):
                        child.mutation_change_weights()
                    if self.choose_with_probability(self.__mutation_add_synapse_chance):
                        child.mutation_add_synapse()
                    if self.choose_with_probability(self.__mutation_add_neuron_chance):
                        child.mutation_add_neuron()

                    # Offspring is added
                    new_population.append(child)

        # New blood and copies of the old generation's best performers are added
        if len(new_population) < self.__population_size:
            # The amounts of each are calculated
            great_performers = int((self.__population_size-len(new_population))/2.0)
            new_performers = (self.__population_size-len(new_population))-great_performers

            # The elements are added
            for i in range(great_performers):
                new_population.append(self.__population[self.__population_size-1-i])

            for i in range(new_performers):
                new_population.append(Network.new(self.__input_amount, self.__output_amount, seed=self.__seed))
                self.__seed += 1

        # Populations are replaced
        self.__population = new_population

    def __calculate_fitnesses(self) -> None:
        """
        Calculates the fitness of every network
        """
        fitnesses = self.__fitness_function(self.__population)
        for i, n in enumerate(self.__population):
            n.set_fitness(max(n.get_fitness()*0.95, fitnesses[i]))

    def __sort_population_by_fitness(self) -> None:
        """
        Sorts the population by fitness, ascendant.
        """
        self.__population = self.__quicksort_by_fitness(self.__population)

    def __separate_species(self) -> None:
        """
        Separates population members into species.
        """
        # Deletes the old species
        self.__symbolic_species.clear()

        # for every network
        for i, n in enumerate(self.__population):
            was_added = False
            # all species representatives are checked
            for s in self.__symbolic_species:
                excess_distance, disjoint_distance, average_weight_difference = \
                    Network.compare(n, self.__population[max(s)])
                distance = (excess_distance * self.__excess_distance_coefficient +
                            disjoint_distance * self.__disjoint_distance_coefficient +
                            average_weight_difference * self.__weights_distance_coefficient)

                # If inter-network distance is small enough, the species adds a new member
                if distance <= self.__species_distance_cap:
                    s.append(i)
                    was_added = True
                    break

            # If no suitable species was found, a new one is created
            if not was_added:
                self.__symbolic_species.append([i])

    def __calculate_shared_fitnesses(self) -> None:
        """
        Calculates the shared fitness of all species members.
        """
        self.__shared_fitness_sums.clear()
        self.__total_shared_fitness = 0.0
        for index, s in enumerate(self.__symbolic_species):
            self.__shared_fitness_sums.append(0.0)
            for i in s:
                shared_fitness = self.__population[i].get_fitness() / len(s)

                # All shared fitnesses are updated
                self.__population[i].set_shared_fitness(shared_fitness)
                self.__shared_fitness_sums[index] += shared_fitness
                self.__total_shared_fitness += shared_fitness

    # GETTERS

    def get_population(self) -> List[Network]:
        """
        Gets the current population of networks.

        :return: A list of networks, the population
        """
        return self.__population

    def get_fitnesses(self) -> List[float]:
        """
        Gets the fitnesses of all population networks.

        :return: A list of all current fitnesses
        """
        fitnesses = []
        for n in self.__population:
            fitnesses.append(n.get_fitness())
        return fitnesses

    def get_best_fitness(self) -> float:
        """
        Gets the current best fitness in the population.

        :return: The fitness of the fittest individual
        """
        return self.__population[len(self.__population)-1].get_fitness()

    def get_maximum_innovation(self) -> int:
        """
        Gets the maximum innovation number (gene count) among the population.

        :return: The largest amount of genes in any individual in the population
        """
        result = 0
        for n in self.__population:
            result = max(result, n.get_innovation())
        return result

    def get_shared_fitness_sums(self) -> List[float]:
        """
        Gets all the shared fitness sums, for all species.

        :return: A list of shared fitness sums
        """
        return self.__shared_fitness_sums

    def get_total_shared_fitness(self) -> float:
        """
        Gets the total shared fitness of the population.

        :return: The total shared fitness
        """
        return self.__total_shared_fitness

    def get_best_network_details(self) -> List[Tuple[int, bool, int, bool, float, bool]]:
        """
        Gets an abstract representation of all connections in the best network of the population.

        Can be used to reproduce the result afterwards.

        :return: Details on the current best network
        """
        best_network = self.__population[len(self.__population)-1]
        return best_network.get_full_details()

    def get_generation(self) -> int:
        """
        Gets the current generation counter value.

        :return: The current generation
        """
        return self.__generation

    # UTILITY METHODS

    @staticmethod
    def __quicksort_by_fitness(population: List[Network]) -> List[Network]:
        """
        Quicksort algorithm for networks.

        :param population: Population to sort
        :return: Sorted population
        """
        lesser = []
        equal = []
        greater = []

        if len(population) <= 1:
            return population
        pivot = population[len(population) - 1]
        for n in population:
            if n.get_fitness() < pivot.get_fitness():
                lesser.append(n)
            elif n.get_fitness() > pivot.get_fitness():
                greater.append(n)
            else:
                equal.append(n)
        return Neat.__quicksort_by_fitness(lesser) + equal + Neat.__quicksort_by_fitness(greater)

    def choose_with_probability(self, probability_percentage: float) -> bool:
        """
        Returns True with the specified percentage probability

        :param probability_percentage: Percentage probability of returning True
        :return: True or False, depending on input and chance
        """
        return self.__generator.uniform(0, 100) <= probability_percentage

    def get_random_float(self) -> float:
        """
        Gets a new random, uniformly distributed, float.

        The range is -2 to 2.

        :return: A new random float
        """
        return self.__generator.uniform(-2.0, 2.0)

    # BUILDER

    @staticmethod
    def builder(input_amount: int, output_amount: int,
                fitness_function: Callable[[List[Network]], List[Union[float, int]]],
                seed: int = None) -> "NeatBuilder":
        """
        Statically gets a builder for the class, makes construction more readable.

        :param input_amount: Input length of the networks to construct
        :param output_amount: Output length of the networks
        :param fitness_function: Fitness function for evaluating the networks
        :param seed: Pseudo-random number generator seed
        :return: A new builder
        """
        return Neat.NeatBuilder(input_amount, output_amount, fitness_function, seed)

    class NeatBuilder:
        """
        Builder class for Neat.
        """

        # CONSTRUCTOR

        def __init__(self, input_amount: int, output_amount: int,
                     fitness_function: Callable[[List[Network]], List[Union[float, int]]], seed: int = None):
            """
            Constructor for the builder.

            Sets default parameters.

            :param input_amount: Input length for all networks.
            :param output_amount: Output length for all networks.
            :param fitness_function: Fitness function for evaluating populations.
            :param seed: Pseudo-random number generator seed
            """
            self.input_amount = input_amount
            self.output_amount = output_amount
            self.fitness_function = fitness_function
            self.seed = seed

            self.population_size = 150
            self.generation_stagnancy_cap = 15
            self.large_species_size = 5

            self.mutation_change_weights_chance = 80.0
            self.mutation_add_synapse_chance = 5.0
            self.mutation_add_neuron_chance = 3.0
            self.no_crossover_chance = 25.0
            self.inter_species_mating_chance = 0.1

            self.excess_distance_coefficient = 1.0
            self.disjoint_distance_coefficient = 1.0
            self.weights_distance_coefficient = 0.4

            self.species_distance_cap = 3.0

        # BUILD METHOD

        def build(self) -> "Neat":
            """
            Constructs a new instance of Neat from the builder's parameters.

            :return: The new Neat instance
            """
            return Neat(self.input_amount, self.output_amount, self.fitness_function, self.population_size,
                        self.generation_stagnancy_cap, self.large_species_size, self.mutation_change_weights_chance,
                        self.mutation_add_synapse_chance, self.mutation_add_neuron_chance, self.no_crossover_chance,
                        self.inter_species_mating_chance, self.excess_distance_coefficient,
                        self.disjoint_distance_coefficient, self.weights_distance_coefficient,
                        self.species_distance_cap, self.seed)

        # SETTERS

        def set_input_amount(self, input_amount: int) -> "Neat.NeatBuilder":
            self.input_amount = input_amount
            return self

        def set_output_amount(self, output_amount: int) -> "Neat.NeatBuilder":
            self.output_amount = output_amount
            return self

        def sef_fitness_function(self, fitness_function: Callable[[List[Network]], List[Union[float, int]]]) \
                -> "Neat.NeatBuilder":
            self.fitness_function = fitness_function
            return self

        def set_seed(self, seed: int) -> "Neat.NeatBuilder":
            self.seed = seed
            return self

        def set_population_size(self, population_size: int = 150) -> "Neat.NeatBuilder":
            self.population_size = population_size
            return self

        def set_generation_stagnancy_cap(self, generation_stagnancy_cap: int = 15) -> "Neat.NeatBuilder":
            self.generation_stagnancy_cap = generation_stagnancy_cap
            return self

        def set_large_species_size(self, large_species_size: int = 5) -> "Neat.NeatBuilder":
            self.large_species_size = large_species_size
            return self

        def set_mutation_change_weights_chance(self,
                                               mutation_change_weights_chance: float = 80.0) -> "Neat.NeatBuilder":
            self.mutation_change_weights_chance = mutation_change_weights_chance
            return self

        def set_mutation_add_synapse_chance(self, mutation_add_synapse_chance: float = 5.0) -> "Neat.NeatBuilder":
            self.mutation_add_synapse_chance = mutation_add_synapse_chance
            return self

        def set_mutation_add_neuron_chance(self, mutation_add_neuron_chance: float = 3.0) -> "Neat.NeatBuilder":
            self.mutation_add_neuron_chance = mutation_add_neuron_chance
            return self

        def set_no_crossover_chance(self, no_crossover_chance: float = 25.0) -> "Neat.NeatBuilder":
            self.no_crossover_chance = no_crossover_chance
            return self

        def set_inter_species_mating_chance(self, inter_species_mating_chance: float = 0.1) -> "Neat.NeatBuilder":
            self.inter_species_mating_chance = inter_species_mating_chance
            return self

        def set_excess_distance_coefficient(self, excess_distance_coefficient: float = 1.0) -> "Neat.NeatBuilder":
            self.excess_distance_coefficient = excess_distance_coefficient
            return self

        def set_disjoint_distance_coefficient(self, disjoint_distance_coefficient: float = 1.0) -> "Neat.NeatBuilder":
            self.disjoint_distance_coefficient = disjoint_distance_coefficient
            return self

        def set_weights_distance_coefficient(self, weights_distance_coefficient: float = 0.4) -> "Neat.NeatBuilder":
            self.weights_distance_coefficient = weights_distance_coefficient
            return self

        def set_species_distance_cap(self, species_distance_cap: float = 3.0) -> "Neat.NeatBuilder":
            self.species_distance_cap = species_distance_cap
            return self
