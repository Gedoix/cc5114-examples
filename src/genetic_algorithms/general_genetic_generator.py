import string

from file_utilities import dir_management as dm

import random

import matplotlib.pyplot as plt


class GeneticGuesser:

    def __init__(self, individuals_amount: int,
                 genes_per_individual: int,
                 gene_alphabet: list,
                 evaluating_function=None,
                 max_possible_fitness: int = None,
                 survivors_percentage: float = 25,
                 mutation_chance_percentage: float = 1,
                 seed: int = None):

        self.__individuals_amount = individuals_amount
        self.__genes_per_individual = genes_per_individual
        self.__gene_alphabet = gene_alphabet

        if evaluating_function is None:
            self.__evaluating_function = self._null_evaluating_function
        else:
            self.__evaluating_function = evaluating_function

        if max_possible_fitness is None:
            self.__max_fitness = self.__genes_per_individual
        else:
            self.__max_fitness = max_possible_fitness

        self.__survivors_percentage = survivors_percentage
        self.__mutation_chance_percentage = mutation_chance_percentage

        self.__random = random.Random()
        if seed is not None:
            self.__random.seed(seed)

        self.__genes = []
        self.__fitness_scores = []

        for i in range(individuals_amount):
            gene = []
            for j in range(genes_per_individual):
                gene.append(gene_alphabet[self.__random.randint(0, len(gene_alphabet) - 1)])
            self.__genes.append(gene)
        pass

    @staticmethod
    def builder() -> 'GeneticGuesser.Builder':
        return GeneticGuesser.Builder()

    def change_evaluation_function(self, fn) -> None:
        self.__evaluating_function = fn
        return

    def change_max_possible_fitness(self, fitness: int) -> None:
        self.__max_fitness = fitness
        return

    @staticmethod
    def _null_evaluating_function(word: list) -> int:
        return len(word)

    def __evaluate(self) -> None:
        self.__fitness_scores = []
        for gene in self.__genes:
            fitness = self.__evaluating_function(gene)
            self.__fitness_scores.append(fitness)
        return

    def is_done(self) -> bool:
        for evaluation in self.__fitness_scores:
            if evaluation == self.__max_fitness:
                return True
        return False

    def get_max_fitness(self) -> int:
        return max(self.__fitness_scores)

    def get_best_individual(self) -> list:
        return self.__genes[self.__fitness_scores.index(self.get_max_fitness())]

    def __select(self) -> None:
        for i in range(int(len(self.__genes) * (1.0 - self.__survivors_percentage / 100.0))):
            index = self.__fitness_scores.index(min(self.__fitness_scores))
            self.__genes.pop(index)
            self.__fitness_scores.pop(index)
        return

    def __reproduce(self) -> None:
        new_genes = []
        for i in range(self.__individuals_amount):
            index_1 = self.__random.randint(0, len(self.__genes) - 1)
            index_2 = self.__random.randint(0, len(self.__genes) - 1)
            if len(self.__genes) != 2:
                while index_1 == index_2:
                    index_2 = self.__random.randint(0, len(self.__genes) - 1)
            elif index_1 == index_2:
                index_2 = 0 if index_1 == 1 else 1
            gene = []
            for j in range(self.__genes_per_individual):
                r = self.__random.uniform(0.0, 100.0 + self.__mutation_chance_percentage)
                if r <= 50:
                    gene.append(self.__genes[index_1][j])
                elif r <= 100:
                    gene.append(self.__genes[index_2][j])
                else:
                    gene.append(self.__gene_alphabet[self.__random.randint(0, len(self.__gene_alphabet) - 1)])

            new_genes.append(gene)
        self.__genes = new_genes
        return

    def generational_step(self) -> bool:
        if len(self.__fitness_scores) == 0:
            self.__evaluate()
        if not self.is_done():
            self.__select()
            self.__reproduce()
            self.__evaluate()
        return self.is_done()

    class Builder:

        def __init__(self):
            self.__individuals_amount = None
            self.__genes_per_individual = None
            self.__gene_alphabet = None

            self.__evaluating_function = GeneticGuesser._null_evaluating_function

            self.__max_fitness = None

            self.__survivors_percentage = 25
            self.__mutation_chance_percentage = 1.0

            self.__seed = None
            return

        def with_individuals(self, amount: int) -> 'GeneticGuesser.Builder':
            self.__individuals_amount = amount
            return self

        def with_genes_amount(self, amount_per_individual: int) -> 'GeneticGuesser.Builder':
            self.__genes_per_individual = amount_per_individual
            return self

        def with_alphabet(self, alphabet: list) -> 'GeneticGuesser.Builder':
            self.__gene_alphabet = alphabet
            return self

        def with_evaluating_function(self, func) -> 'GeneticGuesser.Builder':
            self.__evaluating_function = func
            return self

        def with_max_fitness(self, fitness: int) -> 'GeneticGuesser.Builder':
            self.__max_fitness = fitness
            return self

        def with_survivors(self, percentage: float) -> 'GeneticGuesser.Builder':
            self.__survivors_percentage = percentage
            return self

        def with_mutation_chance(self, percentage: float) -> 'GeneticGuesser.Builder':
            self.__mutation_chance_percentage = percentage
            return self

        def with_seed(self, seed: int) -> 'GeneticGuesser.Builder':
            self.__seed = seed
            return self

        def build(self) -> 'GeneticGuesser':
            if (self.__individuals_amount is not None) and (
                    self.__genes_per_individual is not None) and (
                    self.__gene_alphabet is not None):
                return GeneticGuesser(individuals_amount=self.__individuals_amount,
                                      genes_per_individual=self.__genes_per_individual,
                                      gene_alphabet=self.__gene_alphabet,
                                      evaluating_function=self.__evaluating_function,
                                      max_possible_fitness=self.__max_fitness,
                                      survivors_percentage=self.__survivors_percentage,
                                      mutation_chance_percentage=self.__mutation_chance_percentage,
                                      seed=self.__seed)

            error_message = ""
            if self.__individuals_amount is None:
                error_message += "Amount of Individuals was not specified\n"
            if self.__genes_per_individual is None:
                error_message += "Amount of Genes per Individual was not specified\n"
            if self.__gene_alphabet is None:
                error_message += "Evaluator Function for Individuals was not specified\n"
            raise AttributeError(error_message + "GeneticGuesser Build Failed")


def word_comparator(word_1: list, word_2: list) -> int:
    assert (len(word_1) == len(word_2))
    result = len(word_1)
    for index in range(len(word_1)):
        if word_1[index] != word_2[index]:
            result -= 1
    return result


length_test_plots_saving_directory = './../../plots/word_guessing/length_test'
individuals_test_plots_saving_directory = './../../plots/word_guessing/individuals_test'
survivors_test_plots_saving_directory = './../../plots/word_guessing/survivors_test'
mutation_test_plots_saving_directory = './../../plots/word_guessing/mutation_test'
all_test_plots_saving_directory = './../../plots/word_guessing/individuals_survivors_mutation_test'
alphabet_test_plots_saving_directory = './../../plots/word_guessing/alphabet_test'

english_alphabet = list(string.ascii_lowercase + string.ascii_uppercase)


def main(word_length: int, alphabet: list, individuals_amount: int, survivors_percentage: float,
         mutation_chance_percentage: float, seed: int = None, save_directory: str = None, print_info: bool = True):
    a_random = random.Random()
    if seed is not None:
        a_random.seed(seed + 12345)

    if print_info:
        first_10_alphabet = str(alphabet[0])
        for i in range(min(len(alphabet) - 1, 9)):
            first_10_alphabet += "," + str(alphabet[i + 1])
        first_10_alphabet += "...[" + str(len(alphabet) - 10) + " more elements]" if len(alphabet) > 10 else ""
        print("\nWord of length " + str(word_length) + "\nAnd alphabet: \n    " + first_10_alphabet)

    iteration = 0
    max_fitness_scores = []

    word = []
    for _ in range(word_length):
        word.append(alphabet[a_random.randint(0, len(alphabet) - 1)])

    if print_info:
        first_10_word = str(word[0])
        for i in range(min(len(word) - 1, 9)):
            first_10_word += str(word[i + 1])
        first_10_word += "..." if len(word) > 10 else ""
        print("\nThe target word is:\n    " + str(first_10_word))

    while True:

        initial_iterations = iteration

        guesser = GeneticGuesser.Builder() \
            .with_individuals(individuals_amount) \
            .with_genes_amount(word_length) \
            .with_alphabet(alphabet) \
            .with_evaluating_function(lambda gene: word_comparator(gene, word)) \
            .with_max_fitness(word_length) \
            .with_survivors(survivors_percentage) \
            .with_mutation_chance(mutation_chance_percentage) \
            .with_seed(seed) \
            .build()

        while iteration < initial_iterations + int(0.1*(word_length * len(alphabet))**2) and \
                not guesser.generational_step():
            max_fitness_scores.append(guesser.get_max_fitness())
            iteration += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iteration += 1

        if iteration != initial_iterations + (word_length * len(alphabet) * 10) + 1:
            break
        else:
            seed += 1

    _, ax = plt.subplots()

    ax.plot(range(iteration), max_fitness_scores)
    ax.set_xlim([0, iteration])
    ax.set_ylim([max(min(min(max_fitness_scores), word_length - 100), 0), word_length])

    plt.title("\nGenerational fitness for a genetic word guesser of length '" + str(word_length) +
              "' \namount of gene individuals '" + str(individuals_amount) +
              "' \npercentage of survivors per generation '" + str(survivors_percentage) +
              "' \nand mutation chance per gene '" + str(mutation_chance_percentage) + "'")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness")
    ax.grid(True)

    if save_directory is not None:
        plt.savefig(save_directory + "/plot_word" + str(word_length) + "_individuals" +
                    str(individuals_amount) + "_survivors" + str(survivors_percentage) + "_mutation" +
                    str(mutation_chance_percentage) + ".png", bbox_inches='tight')
    else:
        plt.show()

    print("\nThe solution was found after " + str(iteration) + " iterations of the guesser")

    plt.close()

    return iteration


def plot_result(x: list, y: list, title: str, xlab: str, ylab: str, save_directory: str = None):
    _, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_xlim([0, max(x)])
    ax.set_ylim([max(min(min(y), max(y) - 100), 0), max(y)])

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.grid(True)

    if save_directory is not None:
        plt.savefig(save_directory + "/final_comparison.png", bbox_inches='tight')
    else:
        plt.show()
    return


if __name__ == '__main__':

    points = 20

    binary_alphabet = ['0', '1']

    s = 1

    #   ------  Finding Length  ------

    print("\nRunning length test")

    dm.clear_dir(length_test_plots_saving_directory)

    diffs = []
    iterations = []

    for diff in range(points):
        p = (float(diff) + 1) / points
        diffs.append(100 + int(100*p))
        iterations.append(main(diffs[diff], binary_alphabet, 3*diffs[diff], 25.0, 5.0, seed=s,
                               save_directory=length_test_plots_saving_directory))
        s += 1

    plot_result(diffs, iterations, "Length vs Iterations needed", "Length", "Iterations",
                save_directory=length_test_plots_saving_directory)

    #   ------  Finding Individuals  ------

    print("\nRunning individuals amount test")

    dm.clear_dir(individuals_test_plots_saving_directory)

    diffs = []
    iterations = []

    length = 170

    for diff in range(points):
        p = (float(diff) + 1) / points
        diffs.append(2*length + int(2*length * p))
        iterations.append(main(length, binary_alphabet, diffs[diff], 25.0, 5.0, seed=s,
                               save_directory=individuals_test_plots_saving_directory))
        s += 1

    plot_result(diffs, iterations, "Amount of individuals vs Iterations needed", "Individuals", "Iterations",
                save_directory=individuals_test_plots_saving_directory)

    #   ------  Finding Survivors  ------

    print("\nRunning survivors proportion test")

    dm.clear_dir(survivors_test_plots_saving_directory)

    diffs = []
    iterations = []

    individuals = 400

    for diff in range(points):
        p = (float(diff) + 1) / points
        diffs.append(25.0 + (50.0 * p))
        iterations.append(main(length, binary_alphabet, individuals, diffs[diff], 5.0, seed=s,
                               save_directory=survivors_test_plots_saving_directory))
        s += 1

    plot_result(diffs, iterations, "Survivors proportion vs Iterations needed", "Survivors percentage", "Iterations",
                save_directory=survivors_test_plots_saving_directory)

    #   ------  Finding Mutation Chance  ------

    print("\nRunning mutation chance test")

    dm.clear_dir(mutation_test_plots_saving_directory)

    diffs = []
    iterations = []

    survivors = 26.0

    for diff in range(points):
        p = (float(diff) + 1) / points
        diffs.append(0.5 + (5.0 * p))
        iterations.append(main(length, binary_alphabet, individuals, survivors, diffs[diff], seed=s,
                               save_directory=mutation_test_plots_saving_directory))
        s += 1

    plot_result(diffs, iterations, "Mutation chance percentage vs Iterations needed", "Mutation chance", "Iterations",
                save_directory=mutation_test_plots_saving_directory)

    #   ------  Finding Feature Comparison  ------

    # print("\nRunning all features test")
    #
    # dm.clear_dir(all_test_plots_saving_directory)
    #
    # diffs = []
    # iterations = []
    #
    # for i_1 in range(points):
    #     sub_diffs = []
    #     sub_iterations = []
    #     p = (float(i_1) + 1) / points
    #     individuals = 2*length + int(2*length * p)
    #     for i_2 in range(points):
    #         sub_sub_diffs = []
    #         sub_sub_iterations = []
    #         sub_p = (float(i_2) + 1) / points
    #         survivors = 25.0 + (50.0 * sub_p)
    #         for i_3 in range(points):
    #             sub_sub_p = (float(i_3) + 1) / points
    #             mutation = 0.5 + (5.0 * sub_sub_p)
    #             sub_sub_diffs.append((individuals, survivors, mutation))
    #             sub_sub_iterations.append(main(length, binary_alphabet, individuals, survivors, mutation, seed=s,
    #                                            save_directory=all_test_plots_saving_directory))
    #             s += 1
    #         sub_diffs.append(sub_sub_diffs)
    #         sub_iterations.append(sub_sub_iterations)
    #     diffs.append(sub_diffs)
    #     iterations.append(sub_iterations)
    #
    # plot_result(diffs, iterations, "Complicated stuff vs Iterations needed", "Individuals", "Iterations",
    #             save_directory=all_test_plots_saving_directory)

    #   ------  Testing English Alphabet  ------

    print("\nRunning test on the english alphabet")

    dm.clear_dir(alphabet_test_plots_saving_directory)

    diffs = []
    iterations = []

    for diff in range(points):
        p = (float(diff) + 1) / points
        diffs.append(0 + int(100 * p))
        iterations.append(main(diffs[diff], english_alphabet, 3 * diffs[diff], 25.0, 5.0, seed=s,
                               save_directory=alphabet_test_plots_saving_directory))
        s += 1

    plot_result(diffs, iterations, "Length vs Iterations needed with a full English alphabet", "Length", "Iterations",
                save_directory=alphabet_test_plots_saving_directory)
