from file_utilities import dir_management as dm

import matplotlib.pyplot as plt
from matplotlib.table import Table

from genetic_algorithms.general_genetic_generator import GeneticGuesser


def fitness_evaluator(queens_configuration: list) -> int:
    fitness = sum(range(len(queens_configuration)))
    for i in range(int(len(queens_configuration) / 2)):
        x = queens_configuration[2 * i]
        y = queens_configuration[2 * i + 1]
        for j in range(int((len(queens_configuration)/2)-i-1)):
            j += i+1
            x2 = queens_configuration[2 * j]
            y2 = queens_configuration[2 * j + 1]
            if x == x2 or y == y2 or abs(x-x2) == abs(y-y2):
                fitness -= 1
    return fitness


def print_queens(queens: list) -> None:
    text = "\n"
    queens_2 = list(queens)
    board_size = int(len(queens)/2.0)
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
def plot_queens(queens: list, title: str, file_path_and_name: str = None) -> None:
    queens_2 = list(queens)
    board_size = int(len(queens) / 2)

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

    if file_path_and_name is not None:
        plt.savefig(file_path_and_name, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    return


plots_saving_directory = './../../plots/n_queen/fitness_plots'
boards_saving_directory = './../../plots/n_queen/board_configurations'


def main(board_size: int = 4, survivors_percentage: int = 25, mutation_change_percentage: float = 2.0,
         seed: int = None, save_plots: bool = False, show_all_plots: bool = False, print_info: bool = True):

    queen_amount = board_size

    max_iterations_per_try = (board_size**2)*3+100

    attempt_counter = 1

    while True:
        if print_info:
            print("\nAttempt number "+str(attempt_counter))

        guesser = GeneticGuesser.Builder() \
            .with_individuals(100) \
            .with_genes_amount(2 * queen_amount) \
            .with_alphabet(list(range(board_size))) \
            .with_evaluating_function(lambda gene: fitness_evaluator(gene)) \
            .with_max_fitness(sum(range(2 * queen_amount))) \
            .with_survivors(survivors_percentage) \
            .with_mutation_chance(mutation_change_percentage) \
            .with_seed(seed) \
            .build()

        iterations = 0
        max_fitness_scores = []

        while not (guesser.generational_step() or iterations == max_iterations_per_try-1):
            max_fitness_scores.append(guesser.get_max_fitness())
            iterations += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1

        if print_info:
            state = "success" if (iterations != max_iterations_per_try) else "failure"
            print("\n    Max fitness achieved by attempt " + str(attempt_counter) +
                  " was "+str(guesser.get_max_fitness())+"/"+str(sum(range(2*queen_amount)))
                  + "\n    after " + str(iterations) + " genetic iterations, \n    this was a "+state)

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

        if iterations != max_iterations_per_try:
            queens = guesser.get_best_individual()
            break

        if seed is not None:
            seed += 1

        attempt_counter += 1

    title = "Solution to the " + str(board_size) + "-Queens Problem after " + str(attempt_counter) + \
            " Attempts, " + str(attempt_counter*max_iterations_per_try+iterations) + " Iterations"

    if save_plots:
        plot_queens(queens, title, boards_saving_directory + "/plot_board" + str(board_size) + "_result.png")
    else:
        plot_queens(queens, title, None)

    if print_info:
        print("\nA solution to the N-Queens problem in a " + str(board_size) + " times " + str(board_size) +
              " \nboard was found after " + str(attempt_counter) + " attempts")
        print("\nThe total amount of genetic iterations was " + str(attempt_counter * max_iterations_per_try +
                                                                    iterations))
        print_queens(queens)

    return


if __name__ == '__main__':

    dm.clear_dir(plots_saving_directory)
    dm.clear_dir(boards_saving_directory)

    for board in range(10):
        board *= 4
        board += 4
        main(board_size=board, seed=123456789, save_plots=True, show_all_plots=False, print_info=True)
