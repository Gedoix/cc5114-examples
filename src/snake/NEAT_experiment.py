import random
from random import Random
from typing import List, Optional, Union

from neuroevolution_algorithms.NEAT import Neat
from neuroevolution_algorithms.neat_network import Network
from snake.snake_game import Game, Snake, AI


LOG = {"Experiment": True, "ExperimentAI": True, "FrequentSimulations": False}


class Experiment:

    last_best_score: int
    last_used_seed: int
    last_snakes: List[Snake]
    generator: Random
    seed: Optional[int]
    neat: Neat

    def __init__(self, seed: int = None):
        if LOG["Experiment"]:
            print("[Experiment] Initializing Experiment")

        self.generator = random.Random()

        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.generator.randint(0, 10000000)

        self.generator.seed(self.seed)

        self.last_used_seed = self.seed

        if LOG["Experiment"]:
            print("[Experiment] seed = " + str(self.seed))

        self.neat = Neat.builder(input_amount=10, output_amount=3,
                                 fitness_function=self.fitness_function, seed=self.seed).\
            set_population_size(2000).\
            set_species_distance_cap(0.25).\
            set_mutation_add_neuron_chance(1.0).\
            set_mutation_add_synapse_chance(2.0).\
            set_mutation_change_weights_chance(85.0).\
            build()

        self.snake_game = Game(11)

    def main(self) -> None:
        if LOG["Experiment"]:
            print("[Experiment] Advancing to 1st Generation")

        self.neat.advance_generation()

        max_fitness = self.neat.get_best_fitness()

        if LOG["Experiment"]:
            print("[Experiment] Entering Main Loop")

        stop = 0.0
        while max_fitness <= 100:
            print("\n[Experiment] Generation = " + str(self.neat.get_generation()))
            print("[Experiment] Maximum Fitness of the Generation = " + str(max_fitness))
            print("[Experiment] Compared to a 'stop' value of = " + str(stop))
            print("[Experiment] Maximum Innovation of the Generation = " + str(self.neat.get_maximum_innovation()))
            print("[Experiment] Shared fitness sums = ", self.neat.get_shared_fitness_sums())
            print("[Experiment] Total shared fitness = ", self.neat.get_total_shared_fitness(), "\n")
            if max_fitness > stop:
                stop = max_fitness
                if LOG["FrequentSimulations"] and input("[Experiment] Simulate? (y/n)\n") == "y":
                    n = self.neat.get_population()[-1]
                    self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                         "Generation = " + str(self.neat.get_generation()),
                                         fps=max(4, int(max_fitness / 4)))

            self.neat.advance_generation()
            max_fitness = self.neat.get_best_fitness()

        if input("[Experiment] Simulate? (y/n)\n") == "y":
            n = self.neat.get_population()[-1]
            self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                 "Generation = " + str(self.neat.get_generation()),
                                 fps=max(4, int(max_fitness / 4)))

        print("The best network generated is specified as:\n", str(self.neat.get_best_network_details()))

        if LOG["Experiment"]:
            print("[Experiment] Quitting Experiment")
        self.snake_game.quit()

    def fitness_function(self, population: List[Network]) -> List[Union[float, int]]:
        # The seed changes
        self.last_used_seed += 1

        # Snakes are re-generated
        snakes = []
        for n in population:
            snakes.append(Snake(11, Experiment.ExperimentAI(n)))

        # Metrics are calculated
        scores, times = self.snake_game.simulate(snakes, self.last_used_seed)

        # The fitnesses are calculated
        fitnesses = []
        for i in range(len(scores)):
            f = scores[i]*(1.0 + 1.0/float(times[i]))
            fitnesses.append(f)

        return fitnesses

    class ExperimentAI(AI):

        network: Network

        def __init__(self, network: Network):
            self.network = network

        def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
                   distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
                   distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float,
                   score: float) -> List[bool]:

            result = self.network.calculate([distance_wall_left, distance_wall_front, distance_wall_right,
                                            distance_tail_left, distance_tail_front, distance_tail_right,
                                            distance_fruit_left, distance_fruit_front, distance_fruit_right,
                                             score])

            return result

        def set_network(self, network: Network) -> None:
            self.network = network

        def get_network(self) -> Network:
            return self.network


def main():
    ex = Experiment(3)  # 3 is best seed, 6 was last tested
    ex.main()


if __name__ == '__main__':
    main()
