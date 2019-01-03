import random
from random import Random
from typing import List, Optional

from neuroevolution_algorithms.NEAT import Neat
from neuroevolution_algorithms.network import Network
from snake.snake_game import Game, Snake, AI

snake_game = Game(11, 11)


class Experiment:

    generator: Random
    seed: Optional[int]
    neat: Neat

    def __init__(self, seed: int = None):
        self.seed = seed
        self.neat = Neat(9, 3, self.fitness_function, seed=seed)
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)
        self.last_snakes = []
        self.last_seeds = []

    def main(self) -> None:
        self.neat.advance_generation()
        max_fitness = max(self.neat.get_fitnesses())
        while max_fitness < 100:
            print("Generation = "+str(self.neat.get_generation()))
            print("Fitness so far = "+str(max_fitness))
            sim = input("Simulate? (y/n)")

            if sim == "y":
                n = self.neat.get_population()[-1]
                snake_game.show(Snake(11, 11, Experiment.ExperimentAI(n)), self.last_seeds[-1],
                                "Generation = "+str(self.neat.get_generation()))

            self.neat.advance_generation()

    def fitness_function(self, population: List[Network]) -> List[float, int]:
        self.last_snakes = []
        self.last_seeds = []
        for n in population:
            self.last_snakes.append(Snake(11, 11, Experiment.ExperimentAI(n)))
            if self.seed is not None:
                self.last_seeds.append(self.generator.randint(0, 10000000))
        self.last_snakes, scores, times = snake_game.simulate(self.last_snakes, self.last_seeds)
        return scores

    class ExperimentAI(AI):

        network: Network

        def __init__(self, network: Network):
            self.network = network

        def choose(self, distance_wall_left: int, distance_wall_front: int, distance_wall_right: int,
                   distance_tail_left: int, distance_tail_front: int, distance_tail_right: int,
                   distance_fruit_left: int, distance_fruit_front: int, distance_fruit_right: int) -> List[bool]:
            return self.network.calculate([distance_wall_left, distance_wall_front, distance_wall_right,
                                           distance_tail_left, distance_tail_front, distance_tail_right,
                                           distance_fruit_left, distance_fruit_front, distance_fruit_right])

        def set_network(self, network: Network) -> None:
            self.network = network

        def get_network(self) -> Network:
            return self.network


def main():
    ex = Experiment(0)
    ex.main()


if __name__ == '__main__':
    main()
