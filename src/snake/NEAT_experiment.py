from file_utilities import dir_management as dm

import matplotlib.pyplot as plt

import random
from random import Random
from typing import List, Optional, Union

from neuroevolution_algorithms.NEAT import Neat
from neuroevolution_algorithms.neat_network import Network
from snake.snake_game import Game, Snake, AI


LOG = {"Experiment": True, "ExperimentAI": False, "FrequentSimulations": False}

SEED = 3

BOARD_SIZE = 11
TARGET_SCORE = 35

PLOT = True
SAVE_PLOTS = True

SHOW_RESULT = False
SAVE_RESULT = True

# Directory for saving plots
plots_saving_directory = './../../plots/snake/fitness_plots'

# Directory for saving board configurations
networks_saving_directory = './../../plots/snake/final_networks'


def main():
    ex = Experiment(SEED)
    ex.main()


class Experiment:

    last_best_score: int
    last_used_seed: int
    last_snakes: List[Snake]
    generator: Random
    seed: Optional[int]
    neat: Neat

    def __init__(self, seed: int = None):
        if LOG["Experiment"]:
            print("\n[Experiment] Initializing Experiment")

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

        self.snake_game = Game(BOARD_SIZE)

    def main(self) -> None:
        if LOG["Experiment"]:
            print("[Experiment] Advancing to 1st Generation")

        self.neat.advance_generation()

        max_fitness = self.neat.get_best_fitness()

        max_fitnesses = [max_fitness]

        if LOG["Experiment"]:
            print("[Experiment] Entering Main Loop")

        stop = 0.0
        while max_fitness <= TARGET_SCORE:
            if LOG["Experiment"]:
                print("\n[Experiment] Generation = " + str(self.neat.get_generation()))
                print("[Experiment] Maximum Fitness of the Generation = " + str(max_fitness))
                print("[Experiment] Compared the Previous Recorded Maximum = " + str(stop))
                print("[Experiment] Maximum Innovation of the Generation = " + str(self.neat.get_maximum_innovation()))
                print("[Experiment] Amount of Species = ", len(self.neat.get_shared_fitness_sums()))
                print("[Experiment] Total Shared Fitness = ", self.neat.get_total_shared_fitness(), "\n")

            if max_fitness > stop:
                stop = max_fitness
                if LOG["FrequentSimulations"] and input("[Experiment] Show Simulation? (y/n)\n") == "y":
                    n = self.neat.get_population()[-1]
                    self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                         "Generation = " + str(self.neat.get_generation()),
                                         fps=max(4, int(max_fitness / 4)))

            self.neat.advance_generation()
            max_fitness = self.neat.get_best_fitness()
            max_fitnesses.append(max_fitness)

        if LOG["Experiment"]:
            print("\n[Experiment] Generation = " + str(self.neat.get_generation()))
            print("[Experiment] Maximum Fitness of the Generation = " + str(max_fitness))
            print("[Experiment] Compared to a 'stop' value of = " + str(stop))
            print("[Experiment] Maximum Innovation of the Generation = " + str(self.neat.get_maximum_innovation()))
            print("[Experiment] Shared fitness sums = ", self.neat.get_shared_fitness_sums())
            print("[Experiment] Total shared fitness = ", self.neat.get_total_shared_fitness(), "\n")

        max_fitness = self.neat.get_best_fitness()
        max_fitnesses.append(max_fitness)

        sim = input("[Experiment] Show Simulation? (y/n)\n")
        while sim == "y":
            n = self.neat.get_population()[-1]
            self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                 "Generation = " + str(self.neat.get_generation()),
                                 fps=max(4, int(max_fitness / 4)))
            sim = input("[Experiment] Show Simulation? (y/n)\n")

        if SHOW_RESULT:
            print("The best network generated is specified as:\n", str(self.neat.get_best_network_details()))

        if SAVE_RESULT:
            if LOG["Experiment"]:
                print("[Experiment] Saving Resulting Network")

            dm.clear_dir(networks_saving_directory)

            with open(networks_saving_directory+"/best_network.txt", "w") as text_file:
                text_file.write(str(self.neat.get_best_network_details()))

            if LOG["Experiment"]:
                print("[Experiment] Resulting Network Saved")

        if PLOT:
            if LOG["Experiment"]:
                print("[Experiment] Generating Fitness Plot")

            dm.clear_dir(plots_saving_directory)

            _, ax = plt.subplots()

            ax.plot(range(1, len(max_fitnesses)+1), max_fitnesses)
            ax.set_xlim([0, len(max_fitnesses)+2])
            ax.set_ylim([max(min(min(max_fitnesses), TARGET_SCORE - 100), 0), TARGET_SCORE+5])

            plt.title("Generational fitness for board size " + str(BOARD_SIZE) +
                      " using seed " + str(SEED))
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            ax.grid(True)

            if SAVE_PLOTS:
                if LOG["Experiment"]:
                    print("[Experiment] Saving Fitness Plot")

                name = plots_saving_directory + "/plot_board" + str(BOARD_SIZE)
                name += ".png"
                plt.savefig(name, bbox_inches='tight')

                if LOG["Experiment"]:
                    print("[Experiment] Fitness Plot Saved")
            else:
                if LOG["Experiment"]:
                    print("[Experiment] Showing Fitness Plot")

                plt.show()

            plt.close()

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
            if LOG["ExperimentAI"]:
                print("[ExperimentAI] Initializing AI")
            self.network = network

        def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
                   distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
                   distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float,
                   score: float) -> List[bool]:

            if LOG["ExperimentAI"]:
                print("[ExperimentAI] Calculating Response")

            result = self.network.calculate([distance_wall_left, distance_wall_front, distance_wall_right,
                                            distance_tail_left, distance_tail_front, distance_tail_right,
                                            distance_fruit_left, distance_fruit_front, distance_fruit_right,
                                             score])

            return result

        def set_network(self, network: Network) -> None:
            self.network = network

        def get_network(self) -> Network:
            return self.network


if __name__ == '__main__':
    main()
