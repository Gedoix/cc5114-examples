import random
from random import Random
from typing import List, Optional, Callable, Union

from neuroevolution_algorithms.neat_network import Network


class Neat:

    __generator: Random
    __total_shared_fitness: float
    __shared_fitness_sums: List[float]
    __generation: int
    __species_distance_cap: float
    __weights_distance_coefficient: float
    __disjoint_distance_coefficient: float
    __excess_distance_coefficient: float
    __inter_species_mating_chance: float
    __no_crossover_chance: float
    __mutation_add_neuron_chance: float
    __mutation_add_synapse_chance: float
    __mutation_change_weights_chance: float
    __large_species_size: int
    __generation_stagnancy_cap: int
    __population_size: int
    __fitness_function: Callable[[List[Network]], List[Union[float, int]]]
    __starting_seed: Optional[int]
    __seed: Optional[int]
    __symbolic_species: List[List[int]]
    __population: List[Network]

    def __init__(self, input_amount: int, output_amount: int,
                 fitness_function: Callable[[List[Network]], List[Union[float, int]]],
                 population_size: int = 150, generation_stagnancy_cap: int = 15, large_species_size: int = 5,
                 mutation_change_weights_chance: float = 80.0, mutation_add_synapse_chance: float = 5.0,
                 mutation_add_neuron_chance: float = 3.0, no_crossover_chance: float = 25.0,
                 inter_species_mating_chance: float = 0.1, excess_distance_coefficient: float = 1.0,
                 disjoint_distance_coefficient: float = 1.0, weights_distance_coefficient: float = 0.4,
                 species_distance_cap: float = 3.0, seed: int = None):

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

        self.__starting_seed = seed
        self.__seed = seed
        self.__generator = random.Random()
        if seed is not None:
            self.__generator.seed(seed)

        self.__generation = 0

        self.__population = []
        self.__symbolic_species = []
        self.__shared_fitness_sums = [0.0]
        self.__total_shared_fitness = 0.0

        first_species = []
        for i in range(population_size):
            n = Network(input_amount, output_amount, seed=seed, birth_generation=0)
            seed += 1
            self.__population.append(n)
            first_species.append(i)
        self.__symbolic_species.append(first_species)

    def get_population(self) -> List[Network]:
        return self.__population

    def get_fitnesses(self) -> List[float]:
        fitnesses = []
        for n in self.__population:
            fitnesses.append(n.get_fitness())
        return fitnesses

    def get_shared_fitnesses(self) -> List[float]:
        shared_fitnesses = []
        for n in self.__population:
            shared_fitnesses.append(n.get_shared_fitness())
        return shared_fitnesses

    def advance_generation(self) -> None:
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
        for index, s in enumerate(self.__symbolic_species[:]):
            best = max(s)
            if best < 0.75*self.__population_size and self.__generation - \
                    self.__population[best].get_birth_generation() >= self.__generation_stagnancy_cap:
                self.__total_shared_fitness -= self.__shared_fitness_sums[index]
                self.__shared_fitness_sums[index] = 0.0

    def __reproduce_population(self) -> None:
        new_population = []
        for index, s in enumerate(self.__symbolic_species):
            keep_best = len(s) >= self.__large_species_size
            offspring_amount = round(self.__population_size * (self.__shared_fitness_sums[index] /
                                                               self.__total_shared_fitness)
                                     if self.__total_shared_fitness != 0
                                     else self.__population_size * (1.0 / len(self.__symbolic_species)))

            if keep_best:
                best_i = max(s)
                new_population.append(self.__population[best_i])

            for _ in range(offspring_amount-1 if keep_best else offspring_amount):
                if self.__generator.uniform(0, 100) <= self.__no_crossover_chance:
                    child = self.__population[s[self.__generator.randint(0, len(s) - 1)]].clone(self.__generation)
                else:
                    child_index = s[self.__generator.randint(0, len(s) - 1)]
                    if len(s) == 1 and child_index == 0:
                        continue
                    while child_index == 0:
                        child_index = s[self.__generator.randint(0, len(s) - 1)]
                    child = self.__population[child_index].clone(self.__generation)
                    if self.__generator.uniform(0, 100) <= self.__inter_species_mating_chance:
                        mate_index = self.__generator.randint(0, len(self.__population) - 1)
                        while mate_index < child_index:
                            mate_index = self.__generator.randint(0, len(self.__population) - 1)
                    else:
                        mate_index = s[self.__generator.randint(0, len(s) - 1)]
                        while mate_index < child_index:
                            mate_index = s[self.__generator.randint(0, len(s) - 1)]
                    mate = self.__population[mate_index]
                    child.crossover(mate)

                if self.__generator.uniform(0, 100) <= self.__mutation_change_weights_chance:
                    child.mutation_change_weights()
                if self.__generator.uniform(0, 100) <= self.__mutation_add_synapse_chance:
                    child.mutation_add_synapse()
                if self.__generator.uniform(0, 100) <= self.__mutation_add_neuron_chance:
                    child.mutation_add_neuron()
                new_population.append(child)

        while len(new_population) < self.__population_size:
            child_index = self.__generator.randint(1, len(self.__population) - 1)
            child = self.__population[child_index].clone(self.__generation)
            mate_index = self.__generator.randint(0, len(self.__population) - 1)
            while mate_index < child_index:
                mate_index = self.__generator.randint(0, len(self.__population) - 1)
            mate = self.__population[mate_index]
            child.crossover(mate)
            if self.__generator.uniform(0, 100) <= self.__mutation_change_weights_chance:
                child.mutation_change_weights()
            if self.__generator.uniform(0, 100) <= self.__mutation_add_synapse_chance:
                child.mutation_add_synapse()
            if self.__generator.uniform(0, 100) <= self.__mutation_add_neuron_chance:
                child.mutation_add_neuron()
            new_population.append(child)

        self.__population = new_population

    def __calculate_fitnesses(self) -> None:
        fitnesses = self.__fitness_function(self.__population)
        for i, n in enumerate(self.__population):
            n.set_fitness(fitnesses[i])

    def __sort_population_by_fitness(self) -> None:
        self.__population = self.__quicksort_by_fitness(self.__population)

    def __separate_species(self) -> None:
        self.__symbolic_species.clear()
        for i, n in enumerate(self.__population):
            was_added = False
            for s in self.__symbolic_species:
                excess_distance, disjoint_distance, average_weight_difference = Network.compare(n,
                                                                                                self.__population[s[0]])
                distance = (excess_distance * self.__excess_distance_coefficient +
                            disjoint_distance * self.__disjoint_distance_coefficient +
                            average_weight_difference * self.__weights_distance_coefficient)
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

    def get_generation(self) -> int:
        return self.__generation

    @staticmethod
    def __quicksort_by_fitness(population: List[Network]) -> List[Network]:
        lesser = []
        equal = []
        greater = []

        if len(population) <= 1:
            return population
        pivot = population[0]
        for n in population:
            if n.get_fitness() < pivot.get_fitness():
                lesser.append(n)
            elif n.get_fitness() > pivot.get_fitness():
                greater.append(n)
            else:
                equal.append(n)
        return Neat.__quicksort_by_fitness(lesser) + equal + Neat.__quicksort_by_fitness(greater)

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
