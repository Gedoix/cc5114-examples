import random
from random import Random
from typing import List, Optional, Callable, Union, Tuple

from neuroevolution_algorithms.neat_network import Network


class Neat:

    __fitness_function: Callable[[List[Network]], List[Union[float, int]]]

    __population_size: int
    __generation_stagnancy_cap: int
    __large_species_size: int

    __mutation_change_weights_chance: float
    __mutation_add_synapse_chance: float
    __mutation_add_neuron_chance: float

    __no_crossover_chance: float
    __inter_species_mating_chance: float

    __excess_distance_coefficient: float
    __disjoint_distance_coefficient: float
    __weights_distance_coefficient: float

    __species_distance_cap: float

    __starting_seed: Optional[int]
    __seed: Optional[int]
    __generator: Random

    __generation: int

    __population: List[Network]
    __symbolic_species: List[List[int]]

    __shared_fitness_sums: List[float]
    __total_shared_fitness: float

    def __init__(self, input_amount: int, output_amount: int,
                 fitness_function: Callable[[List[Network]], List[Union[float, int]]],
                 population_size: int = 150, generation_stagnancy_cap: int = 15, large_species_size: int = 5,
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
        :param generation_stagnancy_cap: Cap of stagnancy in generations
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
        self.__fitness_function = fitness_function

        self.__population_size = population_size
        self.__generation_stagnancy_cap = generation_stagnancy_cap
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

        self.__seed = seed
        self.__generator = random.Random()
        if seed is not None:
            self.__generator.seed(seed)

        self.__starting_seed = self.__generator.getstate()[0]

        self.__generation = 0

        self.__population = []
        self.__symbolic_species = []

        self.__shared_fitness_sums = [0.0]
        self.__total_shared_fitness = 0.0

        # The population is initialized
        first_species = []
        for i in range(population_size):
            n = Network.new(input_amount, output_amount, seed=seed)
            seed += 1
            self.__population.append(n)
            first_species.append(i)
        self.__symbolic_species.append(first_species)

    def advance_generation(self) -> None:
        """
        Advances the population by a single generation.

        First it cleans useless species, then reproduces those who are left, re-calculates their fitnesses, sorts
        them, separates species, and finally calculates the shared fitnesses of all networks.
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

    def __remove_useless_species(self) -> None:
        """
        Sets a species as removable by changing it's shared fitness sum to 0.

        Removes large species with under-achieving fitnesses from the population
        """
        for index, s in enumerate(self.__symbolic_species):
            best = max(s)
            if best <= 0.75*self.__population_size and len(self.__shared_fitness_sums) >= self.__large_species_size:
                print("-------------------------Species deleted")
                self.__total_shared_fitness -= self.__shared_fitness_sums[index]
                self.__shared_fitness_sums[index] = 0.0
                for n_index in s:
                    self.__population[n_index].set_shared_fitness(0.0)

    def __reproduce_population(self) -> None:
        new_population = []
        for index, s in enumerate(self.__symbolic_species):
            if self.__shared_fitness_sums[index] != 0.0 or self.__total_shared_fitness == 0.0:
                keep_best = len(s) >= self.__large_species_size
                offspring_amount = int(self.__population_size * (self.__shared_fitness_sums[index] /
                                                                 self.__total_shared_fitness)
                                       if self.__total_shared_fitness != 0
                                       else self.__population_size * (1.0 / len(self.__symbolic_species)))

                if keep_best:
                    best_i = max(s)
                    new_population.append(self.__population[best_i])
                    offspring_amount -= 1

                for _ in range(offspring_amount):
                    if len(s) == 1 or self.choose_with_probability(self.__no_crossover_chance):
                        child = self.__population[s[self.__generator.randint(0, len(s) - 1)]].clone()
                    else:
                        child_index = s[self.__generator.randint(0, len(s) - 1)]
                        child = self.__population[child_index]
                        if self.choose_with_probability(self.__inter_species_mating_chance):
                            mate_index = self.__generator.randint(0, len(self.__population) - 1)
                            while mate_index == child_index or (
                                    self.__population[mate_index].get_shared_fitness() == 0.0 and
                                    not self.__total_shared_fitness == 0.0):
                                mate_index = self.__generator.randint(0, len(self.__population) - 1)
                        else:
                            mate_index = s[self.__generator.randint(0, len(s) - 1)]
                            while mate_index == child_index:
                                mate_index = s[self.__generator.randint(0, len(s) - 1)]
                        mate = self.__population[mate_index]
                        share_disjoints = child.get_fitness() == mate.get_fitness()
                        if child.get_fitness() >= mate.get_fitness():
                            child = child.clone()
                            child.crossover(mate, share_disjoints)
                        else:
                            mate = mate.clone()
                            mate.crossover(child, share_disjoints)
                            child = mate

                    if self.choose_with_probability(self.__mutation_change_weights_chance):
                        child.mutation_change_weights()
                    if self.choose_with_probability(self.__mutation_add_synapse_chance):
                        child.mutation_add_synapse()
                    if self.choose_with_probability(self.__mutation_add_neuron_chance):
                        child.mutation_add_neuron()
                    new_population.append(child)

        if len(new_population) < self.__population_size:
            for i in range(self.__population_size-len(new_population)):
                new_population.append(self.__population[self.__population_size-1-i])

        self.__population = new_population

    def __calculate_fitnesses(self) -> None:
        fitnesses = self.__fitness_function(self.__population)
        for i, n in enumerate(self.__population):
            n.set_fitness(max(n.get_fitness()*0.99, fitnesses[i]))

    def __sort_population_by_fitness(self) -> None:
        self.__population = self.__quicksort_by_fitness(self.__population)

    def __separate_species(self) -> None:
        self.__symbolic_species.clear()
        for i, n in enumerate(self.__population):
            was_added = False
            for s in self.__symbolic_species:
                excess_distance, disjoint_distance, average_weight_difference = \
                    Network.compare(n, self.__population[max(s)])
                distance = (excess_distance * self.__excess_distance_coefficient +
                            disjoint_distance * self.__disjoint_distance_coefficient +
                            average_weight_difference * self.__weights_distance_coefficient)
                # TODO: Annotation
                # print("Distance = ", str(distance))
                if distance <= self.__species_distance_cap:
                    s.append(i)
                    was_added = True
                    break
            if not was_added:
                self.__symbolic_species.append([i])

    def __calculate_shared_fitnesses(self) -> None:
        self.__shared_fitness_sums.clear()
        self.__total_shared_fitness = 0.0
        for index, s in enumerate(self.__symbolic_species):
            self.__shared_fitness_sums.append(0.0)
            for i in s:
                shared_fitness = self.__population[i].get_fitness() / len(s)
                self.__population[i].set_shared_fitness(shared_fitness)
                self.__shared_fitness_sums[index] += shared_fitness
                self.__total_shared_fitness += shared_fitness

    @staticmethod
    def __quicksort_by_fitness(population: List[Network]) -> List[Network]:
        lesser = []
        equal = []
        greater = []

        if len(population) <= 1:
            return population
        pivot = population[len(population)-1]
        for n in population:
            if n.get_fitness() < pivot.get_fitness():
                lesser.append(n)
            elif n.get_fitness() > pivot.get_fitness():
                greater.append(n)
            else:
                equal.append(n)
        return Neat.__quicksort_by_fitness(lesser) + equal + Neat.__quicksort_by_fitness(greater)

    def get_population(self) -> List[Network]:
        return self.__population

    def get_fitnesses(self) -> List[float]:
        fitnesses = []
        for n in self.__population:
            fitnesses.append(n.get_fitness())
        return fitnesses

    def get_best_fitness(self) -> float:
        return self.__population[len(self.__population)-1].get_fitness()

    def get_maximum_innovation(self) -> int:
        result = 0
        for n in self.__population:
            result = max(result, n.get_innovation())
        return result

    def get_shared_fitness_sums(self) -> List[float]:
        return self.__shared_fitness_sums

    def get_total_shared_fitness(self) -> float:
        return self.__total_shared_fitness

    def get_best_network_details(self) -> List[Tuple[int, bool, int, bool, float, bool]]:
        best_network = self.__population[len(self.__population)-1]
        return best_network.get_full_details()

    def get_generation(self) -> int:
        return self.__generation

    def choose_with_probability(self, probability_percentage: float) -> bool:
        return self.__generator.uniform(0, 100) <= probability_percentage

    def get_random_float(self) -> float:
        return self.__generator.uniform(-2.0, 2.0)

    @staticmethod
    def builder(input_amount: int, output_amount: int,
                fitness_function: Callable[[List[Network]], List[Union[float, int]]],
                seed: int = None) -> "NeatBuilder":
        return Neat.NeatBuilder(input_amount, output_amount, fitness_function, seed)

    class NeatBuilder:

        def __init__(self, input_amount: int, output_amount: int,
                     fitness_function: Callable[[List[Network]], List[Union[float, int]]], seed: int = None):
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

        def build(self) -> "Neat":
            return Neat(self.input_amount, self.output_amount, self.fitness_function, self.population_size,
                        self.generation_stagnancy_cap, self.large_species_size, self.mutation_change_weights_chance,
                        self.mutation_add_synapse_chance, self.mutation_add_neuron_chance, self.no_crossover_chance,
                        self.inter_species_mating_chance, self.excess_distance_coefficient,
                        self.disjoint_distance_coefficient, self.weights_distance_coefficient,
                        self.species_distance_cap, self.seed)

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
