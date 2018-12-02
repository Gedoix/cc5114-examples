from file_utilities import dir_management as dm

import matplotlib.pyplot as plt
from matplotlib.table import Table

from genetic_algorithms.genetic_algorithm import GeneticGuesser


def fitness_evaluation(queens_configuration: list) -> int:
    """
    Simple function for evaluating the fitness of a list of queens

    The analogy for this calculation is as follows:

    Imagine the pieces as nodes of a undirected graph, where only the pieces that can attack each other are connected
    by edges, the most edges that the graph can have is the sum of every positive integer from 1 to the amount of queens

    This case scenario could happen if the queen pieces are all in the same column, for example

    The function returns that number, the sum of all positive integers from 1 to the amount of queens, minus the amount
    of actual edges in the graph for the parameter queens_configuration

    Therefore the more actual edges found, the less fitness

    So if all queens were found on the same column, or row, or a diagonal, the function returns 0

    :param queens_configuration: A list of length 2*<the amount of queens>, with their coordinates
    :return: The fitness of the configuration
    """

    # Ideal value
    fitness = sum(range(len(queens_configuration)))
    for i in range(int(len(queens_configuration) / 2)):
        x = queens_configuration[2 * i]
        y = queens_configuration[2 * i + 1]
        for j in range(int((len(queens_configuration)/2)-i-1)):
            j += i+1
            x2 = queens_configuration[2 * j]
            y2 = queens_configuration[2 * j + 1]
            # Edge found
            if x == x2 or y == y2 or abs(x-x2) == abs(y-y2):
                fitness -= 1
    return fitness


def print_queens(queens_configuration: list) -> None:
    """
    Prints a chess board of colors '#' and '@', with the specified queen pieces as '-'

    :param queens_configuration: A list of length 2*<the amount of queens>, with their coordinates
    :return:
    """
    text = "\n"
    queens_2 = list(queens_configuration)
    board_size = int(len(queens_configuration) / 2.0)
    # Rows
    for i in range(board_size):
        # Columns
        for j in range(board_size):
            is_queen = False
            # See if the space is a queen
            for k in range(int(len(queens_2)/2)):
                x = queens_2[2*k]
                y = queens_2[2*k+1]
                if i == x and j == y:
                    is_queen = True
                    queens_2.pop(2*k)
                    queens_2.pop(2*k)
                    break
            if is_queen:
                text += "|-|"
            elif i % 2 == j % 2:
                text += "|#|"
            else:
                text += "|@|"
        text += "\n"
    print(text)
    return


# https://stackoverflow.com/a/10195347/10216044
def plot_queens(queens_configuration: list, title: str, file_path_and_name: str = None) -> None:
    """
    Plots a chess board of colors yellow and white, with the specified queen pieces marked with a 'Q'

    Can save the plot figure if a path is specified, otherwise it's plotted on a window

    :param queens_configuration: A list of length 2*<the amount of queens>, with their coordinates
    :param title: Title of the plot
    :param file_path_and_name: Path for saving the file
    """
    queens_2 = list(queens_configuration)
    board_size = int(len(queens_configuration) / 2)

    _, ax = plt.subplots()
    ax.set_axis_off()

    table = Table(ax, bbox=[0, 0, 1, 1])
    width = 1.0/board_size
    height = 1.0/board_size
    bkg_colors = ['yellow', 'white']

    for i in range(board_size):
        for j in range(board_size):
            is_queen = False

            for k in range(int(len(queens_2)/2)):
                x = queens_2[2*k]
                y = queens_2[2*k+1]
                if i == x and j == y:
                    is_queen = True
                    queens_2.pop(2*k)
                    queens_2.pop(2*k)
                    break

            idx = [j % 2, (j + 1) % 2][i % 2]
            color = bkg_colors[idx]

            if is_queen:
                table.add_cell(i, j, width, height, text="Q", loc='center', facecolor=color)
            else:
                table.add_cell(i, j, width, height, loc='center', facecolor=color)

    for i in range(board_size):
        # Row Labels...
        table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')
        # Column Labels...
        table.add_cell(-1, i, width, height/2, text=i, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(table)

    plt.title(title, y=1.08)

    # Save or plot
    if file_path_and_name is not None:
        plt.savefig(file_path_and_name, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    return


# Directory for saving plots
plots_saving_directory = './../../plots/n_queen/fitness_plots'

# Directory for saving board configurations
boards_saving_directory = './../../plots/n_queen/board_configurations'


def main(board_size: int = 4, seed: int = None, save_plots: bool = False, show_all_plots: bool = False,
         print_info: bool = True) -> None:
    """
    Main function for solving the N-Queens problem

    Constructs a GeneticGuesser that finds an optimal configuration for a board of board_size

    The Genetic guesser is passed fitness_evaluation(...) as it's evaluation function, and default values of
    survivors_percentage and mutation_change_percentage

    A seed may be specified

    The function can generate plots for each attempt at solving the problem if show_all_plots is set to True, otherwise
    only important plots are generated

    Information regarding the execution of the algorithm is printed if print_info is set to True, otherwise it isn't

    The plots can be saved into memory if save_plots is set to True

    :param board_size: Size of the board's side, and also the amount of queens to fin in it
    :param seed: Seed of the experiment
    :param save_plots: Whether the plots should be saved
    :param show_all_plots: Whether the algorithm should plot intermediate attempts
    :param print_info: Whether the algorithm should print to the standard output as it goes
    """
    queen_amount = board_size

    # Default values
    survivors_percentage = 25.0
    mutation_change_percentage = 2.0

    # Cap of iterations for avoiding local optima
    max_iterations_per_try = (board_size**2)*3+100

    attempt_counter = 1

    while True:
        # Some information is printed
        if print_info:
            print("\nAttempt number "+str(attempt_counter))

        # A guesser is constructed
        guesser = GeneticGuesser.Builder() \
            .with_individuals(100) \
            .with_genes_amount(2 * queen_amount) \
            .with_alphabet(list(range(board_size))) \
            .with_evaluation_function(lambda gene: fitness_evaluation(gene)) \
            .with_max_fitness(sum(range(2 * queen_amount))) \
            .with_survivors(survivors_percentage) \
            .with_mutation_chance(mutation_change_percentage) \
            .with_seed(seed) \
            .build()

        iterations = 0
        max_fitness_scores = []

        # A solution is searched for
        while not (guesser.generational_step() or iterations == max_iterations_per_try-1):
            max_fitness_scores.append(guesser.get_max_fitness())
            iterations += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1

        # Information on whether the algorithm was successful
        if print_info:
            state = "success" if (iterations != max_iterations_per_try) else "failure"
            print("\n    Max fitness achieved by attempt " + str(attempt_counter) +
                  " was "+str(guesser.get_max_fitness())+"/"+str(sum(range(2*queen_amount)))
                  + "\n    after " + str(iterations) + " genetic iterations, \n    this was a "+state)

        # If intermediate plots are needed, or success was achieved
        if show_all_plots or ((not show_all_plots) and (iterations != max_iterations_per_try)):

            _, ax = plt.subplots()

            ax.plot(range(iterations), max_fitness_scores)
            ax.set_xlim([0, iterations])
            max_possible_score = sum(range(2 * queen_amount))
            ax.set_ylim([max(min(min(max_fitness_scores), max_possible_score - 100), 0), max_possible_score])

            plt.title("Generational fitness for board size " + str(board_size) +
                      " during attempt " + str(attempt_counter))
            plt.xlabel("Number of Iterations")
            plt.ylabel("Fitness")
            ax.grid(True)

            if save_plots:
                name = plots_saving_directory + "/plot_board" + str(board_size)
                name += "_attempt" + str(attempt_counter) + ".png" if show_all_plots else ".png"
                plt.savefig(name, bbox_inches='tight')
            else:
                plt.show()

            plt.close()

        # If successful, the result is saved and the loop is exited
        if iterations != max_iterations_per_try:
            queens = guesser.get_best_individual()
            break

        # If not the GeneticGuesser is reset with a different seed
        if seed is not None:
            seed += 1

        attempt_counter += 1

    # A final plot is generated with the configuration found
    title = "Solution to the " + str(board_size) + "-Queens Problem after " + str(attempt_counter) + \
            " Attempts, " + str(attempt_counter*max_iterations_per_try+iterations) + " Iterations"

    # Either saved or displayed
    if save_plots:
        plot_queens(queens, title, boards_saving_directory + "/plot_board" + str(board_size) + "_result.png")
    else:
        plot_queens(queens, title, None)

    # Final information and a and the configuration found are printed
    if print_info:
        print("\nA solution to the N-Queens problem in a " + str(board_size) + " times " + str(board_size) +
              " \nboard was found after " + str(attempt_counter) + " attempts")
        print("\nThe total amount of genetic iterations was " + str(attempt_counter * max_iterations_per_try +
                                                                    iterations))
        print_queens(queens)

    return


if __name__ == '__main__':
    """
    Execution of the file
    
    Runs main for 10 different board sizes
    
    Saves the plots
    """

    dm.clear_dir(plots_saving_directory)
    dm.clear_dir(boards_saving_directory)

    for board in range(10):
        board *= 4
        board += 4
        main(board_size=board, seed=123456789, save_plots=True, show_all_plots=False, print_info=True)
