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
    text = "\nThe Board is:\n"
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
    print("Queens is "+str(queens_2))


def main(board_size: int = 29, max_iterations_per_size: int = 1000):

    greatest = 1

    queens = []

    for queen_amount in range(int(board_size)-1):

        queen_amount = board_size-queen_amount

        guesser = GeneticGuesser(100, 2 * queen_amount,
                                 alphabet=list(range(board_size)),
                                 seed=1234567, survivors_amount=5)

        guesser.set_evaluation_function(fitness_evaluator)
        guesser.set_max_possible_fitness(sum(range(2*queen_amount)))

        iterations = 0
        max_fitness_scores = []

        while not (guesser.generational_step() or iterations == max_iterations_per_size-1):
            max_fitness_scores.append(guesser.get_max_fitness())
            iterations += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1

        state = "success" if (iterations != max_iterations_per_size-1) else "failure"
        print("Max fitness achieved of "+str(guesser.get_max_fitness())+"/"+str(sum(range(2*queen_amount)))
              + " this was a "+state)

        _, ax = plt.subplots()

        ax.plot(range(iterations), max_fitness_scores, 'o')
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, sum(range(2*queen_amount))])

        plt.title("Fitness of each generation for queen amount "+str(queen_amount)+" of "+str(int(board_size)))
        ax.grid(True)
        plt.show()

        if iterations != max_iterations_per_size-1:
            greatest = queen_amount
            queens = guesser.get+_best_word()
            break

    print("\nThe maximum amount of queens that can \n fit a"
          " "+str(board_size)+" times "+str(board_size)+" board \n was found to be "+str(greatest))

    print_queens(queens, board_size)


if __name__ == '__main__':
    main()
