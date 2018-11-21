import matplotlib.pyplot as plt

from genetic_algorithms.general_genetic_generator import GeneticWordGuesser


def fitness_evaluator(queens_configuration: list) -> int:
    errors = 0
    for i in range(int(len(queens_configuration) / 2)):
        x = queens_configuration[2 * i]
        y = queens_configuration[2 * i + 1]
        for j in range(int((len(queens_configuration) - 2 * i) / 2)):
            x2 = queens_configuration[2 * j]
            y2 = queens_configuration[2 * j + 1]
            if x == x2 or y == y2 or abs(x-x2) == abs(y-y2):
                errors += 1
    errors = sum(range(len(queens_configuration))) - errors
    return errors


def main(board_size: int = 4, max_iterations_per_size: int = 1000):

    for queen_amount in range(board_size):

        guesser = GeneticWordGuesser(100, 2*queen_amount,
                                     alphabet=list(range(board_size)),
                                     seed=1234567, survivors_amount=5)

        guesser.set_evaluation_function(fitness_evaluator)
        guesser.set_max_possible_fitness(sum(range(2*queen_amount)))

        iterations = 0
        max_fitness_scores = []

        while not (guesser.generational_step() or iterations == max_iterations_per_size-1):
            print(guesser.get_max_fitness())
            max_fitness_scores.append(guesser.get_max_fitness())
            iterations += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1

        _, ax = plt.subplots()

        ax.plot(range(iterations), max_fitness_scores, 'o')
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, sum(range(2*queen_amount))])

        plt.title("Maximum Fitness of each generation for a genetic guesser of a "
                  "word of length"+str(sum(range(2*board_size*board_size))))
        ax.grid(True)
        plt.show()

        return queen_amount


if __name__ == '__main__':
    main()
