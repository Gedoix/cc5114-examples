from file_utilities import dir_management as dm

import matplotlib.pyplot as plt

import random
from random import Random
from typing import List, Optional, Union

from neuroevolution_algorithms.NEAT import Neat
from neuroevolution_algorithms.neat_network import Network
from snake.snake_game import Game, Snake, AI


# Logging
LOG = {"Experiment": True, "ExperimentAI": False, "FrequentSimulations": False}

# Pseudo-random seed
SEED = 3

# Game parameters
BOARD_SIZE = 11
TARGET_SCORE = 35

# Plotting fitnesses
PLOT = True
SAVE_PLOTS = True

# Showing results
SHOW_RESULT = False
SAVE_RESULT = True

# Directory for saving plots
plots_saving_directory = './../../plots/snake/fitness_plots'

# Directory for saving board configurations
networks_saving_directory = './../../plots/snake/final_networks'


def main():
    """
    Main function of the experiment.

    Creates a new experiment with the parameters above and runs it.
    """
    ex = Experiment(SEED)
    ex.main()


class Experiment:
    """
    Class container of a full experiment.

    Ends when the target score is achieved by any Neat network.
    """

    last_best_score: int
    last_used_seed: int
    last_snakes: List[Snake]
    generator: Random
    seed: Optional[int]
    neat: Neat

    # CONSTRUCTOR

    def __init__(self, seed: int = None):
        if LOG["Experiment"]:
            print("\n[Experiment] Initializing Experiment")

        # The generator is initialized
        self.generator = random.Random()

        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.generator.randint(0, 10000000)

        self.generator.seed(self.seed)

        self.last_used_seed = self.seed

        if LOG["Experiment"]:
            print("[Experiment] Running with seed = " + str(self.seed))

        # A Neat instance is built
        self.neat = Neat.builder(input_amount=10, output_amount=3,
                                 fitness_function=self.fitness_function, seed=self.seed).\
            set_population_size(2000).\
            set_species_distance_cap(0.25).\
            set_mutation_add_neuron_chance(1.0).\
            set_mutation_add_synapse_chance(2.0).\
            set_mutation_change_weights_chance(85.0).\
            build()

        # A Game is created
        self.snake_game = Game(BOARD_SIZE)

    # MAIN BEHAVIOUR

    def main(self) -> None:
        """
        Main method for running the experiment.

        Advances Neat's generations until the target score is reached.

        May save the resulting plots and network if specified by the script constants.
        """
        if LOG["Experiment"]:
            print("[Experiment] Advancing to 1st Generation")

        # Mandatory first generation advancement
        self.neat.advance_generation()

        # Metrics are initialized
        max_fitness = self.neat.get_best_fitness()
        max_fitnesses = [max_fitness]

        if LOG["Experiment"]:
            print("[Experiment] Entering Main Loop")

        # The main loop is entered
        stop = 0.0
        while max_fitness <= TARGET_SCORE:
            # Metrics of the last generation are checked and shared
            if LOG["Experiment"]:
                print("\n[Experiment] Generation = " + str(self.neat.get_generation()))
                print("[Experiment] Maximum Fitness of the Generation = " + str(max_fitness))
                print("[Experiment] Compared the Previous Recorded Maximum = " + str(stop))
                print("[Experiment] Maximum Innovation of the Generation = " + str(self.neat.get_maximum_innovation()))
                print("[Experiment] Amount of Species = ", len(self.neat.get_shared_fitness_sums()))
                print("[Experiment] Total Shared Fitness = ", self.neat.get_total_shared_fitness(), "\n")

            # If an improvement is found, the game may be simulated
            if max_fitness > stop:
                stop = max_fitness
                if LOG["FrequentSimulations"] and input("[Experiment] Show Simulation? (y/n)\n") == "y":
                    n = self.neat.get_population()[-1]
                    self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                         "Generation = " + str(self.neat.get_generation()),
                                         fps=max(4, int(max_fitness / 4)))

            # Generation advancement
            self.neat.advance_generation()
            max_fitness = self.neat.get_best_fitness()
            max_fitnesses.append(max_fitness)

        # If the target was passed, metrics are consulted
        if LOG["Experiment"]:
            print("\n[Experiment] Generation = " + str(self.neat.get_generation()))
            print("[Experiment] Maximum Fitness of the Generation = " + str(max_fitness))
            print("[Experiment] Compared to a 'stop' value of = " + str(stop))
            print("[Experiment] Maximum Innovation of the Generation = " + str(self.neat.get_maximum_innovation()))
            print("[Experiment] Shared fitness sums = ", self.neat.get_shared_fitness_sums())
            print("[Experiment] Total shared fitness = ", self.neat.get_total_shared_fitness(), "\n")

        # Metrics are updated again
        max_fitness = self.neat.get_best_fitness()
        max_fitnesses.append(max_fitness)

        # A simulation of the result can be shown if the user wants to
        sim = input("[Experiment] Show Simulation? (y/n)\n")
        while sim == "y":
            n = self.neat.get_population()[-1]
            self.snake_game.show(Snake(11, Experiment.ExperimentAI(n)), self.last_used_seed,
                                 "Generation = " + str(self.neat.get_generation()),
                                 fps=max(4, int(max_fitness / 4)))
            sim = input("[Experiment] Show Simulation? (y/n)\n")

        # The resulting network may be printed
        if SHOW_RESULT:
            print("The best network generated is specified as:\n", str(self.neat.get_best_network_details()))

        # The resulting network may be saved
        if SAVE_RESULT:
            if LOG["Experiment"]:
                print("[Experiment] Saving Resulting Network")

            # Previous saves are removed
            dm.clear_dir(networks_saving_directory)

            # A .txt is generated
            with open(networks_saving_directory+"/best_network.txt", "w") as text_file:
                text_file.write(str(self.neat.get_best_network_details()))

            if LOG["Experiment"]:
                print("[Experiment] Resulting Network Saved")

        # A plot of fitnesses may be created
        if PLOT:
            if LOG["Experiment"]:
                print("[Experiment] Generating Fitness Plot")

            # The plot is generated in matplotlib
            _, ax = plt.subplots()

            ax.plot(range(1, len(max_fitnesses)+1), max_fitnesses)
            ax.set_xlim([0, len(max_fitnesses)+2])
            ax.set_ylim([max(min(min(max_fitnesses), TARGET_SCORE - 100), 0), TARGET_SCORE+5])

            plt.title("Generational fitness for board size " + str(BOARD_SIZE) +
                      " using seed " + str(SEED))
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            ax.grid(True)

            # The plot may be saved to memory
            if SAVE_PLOTS:
                if LOG["Experiment"]:
                    print("[Experiment] Saving Fitness Plot")

                # Previous saves are removed
                dm.clear_dir(plots_saving_directory)

                name = plots_saving_directory + "/plot_board" + str(BOARD_SIZE)
                name += ".png"

                # A new .png is saved
                plt.savefig(name, bbox_inches='tight')

                if LOG["Experiment"]:
                    print("[Experiment] Fitness Plot Saved")
            # Otherwise the plot is displayed
            else:
                if LOG["Experiment"]:
                    print("[Experiment] Showing Fitness Plot")

                plt.show()

            plt.close()

        if LOG["Experiment"]:
            print("[Experiment] Quitting Experiment")

        # The experiment ends
        self.snake_game.quit()

    # NEAT'S FITNESS FUNCTION

    def fitness_function(self, population: List[Network]) -> List[Union[float, int]]:
        """
        Fitness function for passing to Neat.

        Calculates fitnesses as a function of score and the time taken to achieve it.

        :param population: Population to evaluate
        :return: A list of all calculated fitnesses
        """
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

    # SNAKE AI

    class ExperimentAI(AI):
        """
        Snake AI for the experiment.
        """

        network: Network

        # CONSTRUCTOR
        def __init__(self, network: Network):
            """
            Constructor for the AI.

            Sets up the network for decision taking.

            :param network: Network to use.
            """
            if LOG["ExperimentAI"]:
                print("[ExperimentAI] Initializing AI")
            self.network = network

        # MAIN BEHAVIOUR
        def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
                   distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
                   distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float,
                   score: float) -> List[bool]:
            """
            Chooses a direction for the snake based on the network's results.

            :param distance_wall_left: Distance to the wall on the left
            :param distance_wall_front: Distance to the wall on the front
            :param distance_wall_right: Distance to the wall on the right
            :param distance_tail_left: Distance to a tail segment on the left
            :param distance_tail_front: Distance to a tail segment on the front
            :param distance_tail_right: Distance to a tail segment on the right
            :param distance_fruit_left: Distance to a fruit on the left
            :param distance_fruit_front: Distance to a fruit on the front
            :param distance_fruit_right: Distance to a fruit to the right
            :param score: Current normalized score (snake length)
            :return: The direction the snake should follow
            """
            if LOG["ExperimentAI"]:
                print("[ExperimentAI] Calculating Response")

            result = self.network.calculate([distance_wall_left, distance_wall_front, distance_wall_right,
                                            distance_tail_left, distance_tail_front, distance_tail_right,
                                            distance_fruit_left, distance_fruit_front, distance_fruit_right,
                                             score])

            return result


# EXECUTION
if __name__ == '__main__':
    main()
