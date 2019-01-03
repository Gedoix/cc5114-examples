import random
from random import Random
from typing import List, Optional, Union

from neuroevolution_algorithms.NEAT import Neat
from neuroevolution_algorithms.neat_network import Network
from snake.snake_game import Game, Snake, AI


LOG = {"Experiment": True, "ExperimentAI": True}


class Experiment:

    last_used_seed: int
    last_snakes: List[Snake]
    generator: Random
    seed: Optional[int]
    neat: Neat

    def __init__(self, seed: int = None):
        if LOG["Experiment"]:
            print("[Experiment] Initializing Experiment")
        self.seed = seed
        self.neat = Neat.builder(9, 3, self.fitness_function, seed).\
            set_species_distance_cap(2.0).\
            build()
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)
        self.snake_game = Game(11, 11)
        self.last_snakes = []
        self.last_used_seed = self.generator.getstate()[0]

    def main(self) -> None:
        if LOG["Experiment"]:
            print("[Experiment] Advancing to 1st Generation")
        self.neat.advance_generation()
        max_fitness = max(self.neat.get_fitnesses())
        stop = 0
        if LOG["Experiment"]:
            print("[Experiment] Entering Main Loop")
        while max_fitness <= 100:
            print("Generation = " + str(self.neat.get_generation()))
            print("Maximum Fitness of the Generation = " + str(max_fitness))
            if max_fitness >= stop:
                sim = input("Simulate? (y/n)\n")

                if sim == "y":
                    n = self.neat.get_population()[-1]
                    self.snake_game.show(Snake(11, 11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                         "Generation = " + str(self.neat.get_generation()), fps=3)
                stop = max_fitness+1

            print("Shared fitness sums = ", self.neat.get_shared_fitness_sums())
            print("Total shared fitness = ", self.neat.get_total_shared_fitness())
            self.neat.advance_generation()
            max_fitness = max(self.neat.get_fitnesses())
        if LOG["Experiment"]:
            print("[Experiment] Quitting Experiment")
        self.snake_game.quit()

    def fitness_function(self, population: List[Network]) -> List[Union[float, int]]:
        self.last_snakes.clear()
        self.last_used_seed += 1
        for n in population:
            self.last_snakes.append(Snake(11, 11, Experiment.ExperimentAI(n)))
        self.last_snakes, scores, times = self.snake_game.simulate(self.last_snakes, self.last_used_seed)

        return scores

    class ExperimentAI(AI):

        network: Network

        def __init__(self, network: Network):
            self.network = network

        def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
                   distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
                   distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float) -> List[bool]:

            result = self.network.calculate([distance_wall_left, distance_wall_front, distance_wall_right,
                                            distance_tail_left, distance_tail_front, distance_tail_right,
                                            distance_fruit_left, distance_fruit_front, distance_fruit_right])

            return result

        def set_network(self, network: Network) -> None:
            self.network = network

        def get_network(self) -> Network:
            return self.network


def main():
    ex = Experiment(0)
    ex.main()


if __name__ == '__main__':
    main()
