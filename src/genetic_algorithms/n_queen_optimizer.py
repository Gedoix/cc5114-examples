import os
import stat
import shutil
import time

import matplotlib.pyplot as plt

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


def print_queens(queens: list, board_size: int) -> None:
    text = "\n"
    queens_2 = queens
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


# http://stackoverflow.com/questions/1889597/deleting-directory-in-python
def _remove_readonly(fn, path_, excinfo):
    # Handle read-only files and directories
    if fn is os.rmdir:
        os.chmod(path_, stat.S_IWRITE)
        os.rmdir(path_)
    elif fn is os.remove:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)


def force_remove_file_or_symlink(path_):
    try:
        os.remove(path_)
    except OSError:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)


# Code from shutil.rmtree()
def is_regular_dir(path_):
    try:
        mode = os.lstat(path_).st_mode
    except os.error:
        mode = 0
    return stat.S_ISDIR(mode)


def clear_dir(path_):
    if is_regular_dir(path_):
        # Given path is a directory, clear its content
        for name in os.listdir(path_):
            full_path = os.path.join(path_, name)
            if is_regular_dir(full_path):
                shutil.rmtree(full_path, onerror=_remove_readonly)
            else:
                force_remove_file_or_symlink(full_path)
    else:
        # Given path is a file or a symlink.
        # Raise an exception here to avoid accidentally clearing the content
        # of a symbolic linked directory.
        raise OSError("Cannot call clear_dir() on a symbolic link")


def main(board_size: int = 4, survivors_percentage: int = 25, mutation_change_percentage: float = 2.0,
         seed: int = None, save_figures: bool = True):

    queen_amount = board_size

    max_iterations_per_try = (board_size**2)*3+100

    attempt_counter = 1

    save_directory = './../../plots/n_queen'

    time.sleep(1.0)

    while True:
        print("\nAttempt number "+str(attempt_counter))

        guesser = GeneticGuesser(100, 2 * queen_amount,
                                 alphabet=list(range(board_size)),
                                 seed=seed, survivors_percentage=survivors_percentage,
                                 mutation_chance_percentage=mutation_change_percentage)

        guesser.set_evaluation_function(fitness_evaluator)
        guesser.set_max_possible_fitness(sum(range(2*queen_amount)))

        iterations = 0
        max_fitness_scores = []

        while not (guesser.generational_step() or iterations == max_iterations_per_try-1):
            max_fitness_scores.append(guesser.get_max_fitness())
            iterations += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1

        state = "success" if (iterations != max_iterations_per_try) else "failure"
        print("\n    Max fitness achieved by attempt " + str(attempt_counter) +
              " was "+str(guesser.get_max_fitness())+"/"+str(sum(range(2*queen_amount)))
              + "\n    after " + str(iterations) + " genetic iterations, \n    this was a "+state)

        _, ax = plt.subplots()

        ax.plot(range(iterations), max_fitness_scores)
        ax.set_xlim([0, iterations])
        max_possible_score = sum(range(2*queen_amount))
        ax.set_ylim([max(min(min(max_fitness_scores),  max_possible_score-100), 0), max_possible_score])

        plt.title("Generational fitness for board size "+str(board_size) +
                  " during attempt " + str(attempt_counter))
        plt.xlabel("Number of Iterations")
        plt.ylabel("Fitness")
        ax.grid(True)

        if save_figures:
            plt.savefig(save_directory+"/plot_board"+str(board_size)+"_attempt"+str(attempt_counter)+".png",
                        bbox_inches='tight')
        else:

            plt.show()

        plt.close()

        if iterations != max_iterations_per_try:
            queens = guesser.get_best_word()
            break

        if seed is not None:
            seed += 1

        attempt_counter += 1

    print("\nA solution to the N-Queens problem in a " + str(board_size)+" times "+str(board_size) +
          " \nboard was found after " + str(attempt_counter) + " attempts")

    print("\nThe total amount of genetic iterations was "+str(attempt_counter*max_iterations_per_try+iterations))

    print_queens(queens, board_size)


if __name__ == '__main__':

    clear_dir('./../../plots/n_queen')

    for board in range(46):
        board += 4
        main(board_size=board, seed=123456789)
