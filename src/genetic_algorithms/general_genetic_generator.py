import string

from file_utilities import dir_management as dm

import random

import matplotlib.pyplot as plt


class GeneticGuesser:
    """
    Genetic algorithm for constructing arbitrary arrays of genes based on a fitness function that evaluates their
    usefulness
    """

    def __init__(self, individuals_amount: int,
                 genes_per_individual: int,
                 gene_alphabet: list,
                 evaluation_function=None,
                 max_possible_fitness: int = None,
                 survivors_percentage: float = 25,
                 mutation_chance_percentage: float = 1,
                 seed: int = None):
        """
        Constructor for a genetic guesser, sets up default values for optional arguments that aren't passed

        Mainly supposed to be used by the Builder inner class

        :param individuals_amount: Amount of individuals to generate each generation
        :param genes_per_individual: Length of the gene array of each individual
        :param gene_alphabet: Alphabet from which to pick genes at random, the picker function selects indexes of this
        array uniformly, therefore any distribution can be simulated by altering the amount of repetitions of each
        element within the array
        :param evaluation_function: Function that evaluates a list of genes and returns it's fitness, where a bigger
        positive integer is better
        :param max_possible_fitness: The maximum fitness the algorithm looks for, stopping it's evolution when found,
        may be the biggest integer possible in the case that the optimum result is unknown
        :param survivors_percentage: The percentage of the individuals population that survives the selection process,
        only those with the highest fitness can be among this group
        :param mutation_chance_percentage: The chance of any gene in a newly generated offspring of being randomized
        from scratch, represented as a percentage float number
        :param seed: The seed for the guesser's random generator, has deterministic results
        """
        self.__individuals_amount = individuals_amount
        self.__genes_per_individual = genes_per_individual
        self.__gene_alphabet = gene_alphabet

        # Setting a default null evaluation function
        if evaluation_function is None:
            self.__evaluation_function = self._null_evaluating_function
        else:
            self.__evaluation_function = evaluation_function

        # Setting a default maximum fitness
        if max_possible_fitness is None:
            self.__max_fitness = self.__genes_per_individual
        else:
            self.__max_fitness = max_possible_fitness

        self.__survivors_percentage = survivors_percentage
        self.__mutation_chance_percentage = mutation_chance_percentage

        # Setting the seed if available
        self.__random = random.Random()
        if seed is not None:
            self.__random.seed(seed)

        self.__population = []
        self.__fitness_scores = []

        # First generation of individuals
        for i in range(individuals_amount):
            individual = []
            for j in range(genes_per_individual):
                individual.append(gene_alphabet[self.__random.randint(0, len(gene_alphabet) - 1)])
            self.__population.append(individual)
        return

    @staticmethod
    def builder() -> 'GeneticGuesser.Builder':
        """
        Static method that creates a builder for this class, meant to replace the constructor as better syntax

        :return: A builder for the class
        """
        return GeneticGuesser.Builder()

    def change_evaluation_function(self, new_function) -> None:
        """
        Allows changing the evaluation function during execution

        :param new_function: New evaluation function
        """
        self.__evaluation_function = new_function
        return

    def change_max_possible_fitness(self, fitness: int) -> None:
        """
        Allows changing the maximum fitness the algorithm looks for during execution

        :param fitness: New maximum fitness
        """
        self.__max_fitness = fitness
        return

    @staticmethod
    def _null_evaluating_function(individual: list) -> int:
        """
        Null default evaluation function, always returns the length of an individual's gene array

        :param individual: An individual
        """
        return len(individual)

    def __evaluate(self) -> None:
        """
        Runs the evaluation function across all individuals in the population, generating their fitness scores and
        storing them
        """
        self.__fitness_scores = []
        for gene in self.__population:
            fitness = self.__evaluation_function(gene)
            self.__fitness_scores.append(fitness)
        return

    def is_done(self) -> bool:
        """
        Checks whether the maximum fitness possible has been achieved

        :return: True if the algorithm was successful, False if not
        """
        return self.get_max_fitness() == self.__max_fitness

    def get_max_fitness(self) -> int:
        """
        Finds the maximum fitness generated up to now in the population

        :return: The maximum fitness in the population
        """
        return max(self.__fitness_scores)

    def get_best_individual(self) -> list:
        """
        Finds the individual with the best fitness from within the population

        :return: The fittest individual in the population
        """
        return self.__population[self.__fitness_scores.index(self.get_max_fitness())]

    def __select(self) -> None:
        """
        Selects a percentage of the population with the best fitness scores, deleting the rest, this percentage is
        specified upon construction
        """
        for i in range(int(len(self.__population) * (1.0 - self.__survivors_percentage / 100.0))):
            index = self.__fitness_scores.index(min(self.__fitness_scores))
            self.__population.pop(index)
            self.__fitness_scores.pop(index)
        return

    def __reproduce(self) -> None:
        """
        Generates new individuals from the genes of the survivors of the __select() function, replacing the population
        up to max again

        Each new individual can have some of their genes mutated at random with a specific chance specified upon
        construction, this mutation manifests in the generation of said genes randomly from scratch
        """
        new_individuals = []
        for i in range(self.__individuals_amount):
            index_1 = self.__random.randint(0, len(self.__population) - 1)
            index_2 = self.__random.randint(0, len(self.__population) - 1)
            # In case there's many individuals
            if len(self.__population) != 2:
                while index_1 == index_2:
                    index_2 = self.__random.randint(0, len(self.__population) - 1)
            # In case there's only two individuals from which produce offspring
            elif index_1 == index_2:
                index_2 = 0 if index_1 == 1 else 1
            # A new individual is generated
            individual = []
            for j in range(self.__genes_per_individual):
                r = self.__random.uniform(0.0, 100.0 + self.__mutation_chance_percentage)
                if r <= 50:
                    individual.append(self.__population[index_1][j])
                elif r <= 100:
                    individual.append(self.__population[index_2][j])
                # Mutation chance
                else:
                    individual.append(self.__gene_alphabet[self.__random.randint(0, len(self.__gene_alphabet) - 1)])
            # The individual is added
            new_individuals.append(individual)
        self.__population = new_individuals
        return

    def generational_step(self) -> bool:
        """
        Automatically executes a full generation's methods

        :return: True if the maximum fitness possible was found, False if not
        """
        if len(self.__fitness_scores) == 0:
            self.__evaluate()
        if not self.is_done():
            self.__select()
            self.__reproduce()
            self.__evaluate()
        return self.is_done()

    class Builder:
        """
        Builder class for a GeneticGuesser, allows for a better, more clear and meaningful syntax when constructing a
        new guesser
        """

        def __init__(self):
            """
            Constructor that initializes default values for a GeneticGuesser's constructor
            """
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
            """
            Obligatory parameter setter

            :param amount: Amount of individuals to generate each generation
            :return: The current Builder instance
            """
            self.__individuals_amount = amount
            return self

        def with_genes_amount(self, amount_per_individual: int) -> 'GeneticGuesser.Builder':
            """
            Obligatory parameter setter

            :param amount_per_individual: Length of the gene array of each individual
            :return: The current Builder instance
            """
            self.__genes_per_individual = amount_per_individual
            return self

        def with_alphabet(self, alphabet: list) -> 'GeneticGuesser.Builder':
            """
            Obligatory parameter setter

            :param alphabet: Alphabet from which to pick genes at random, the picker function selects indexes of this
            array uniformly, therefore any distribution can be simulated by altering the amount of repetitions of each
            element within the array
            :return: The current Builder instance
            """
            self.__gene_alphabet = alphabet
            return self

        def with_evaluation_function(self, func) -> 'GeneticGuesser.Builder':
            """
            Optional parameter setter

            :param func: Function that evaluates a list of genes and returns it's fitness, where a bigger
            positive integer is better
            :return: The current Builder instance
            """
            self.__evaluating_function = func
            return self

        def with_max_fitness(self, fitness: int) -> 'GeneticGuesser.Builder':
            """
            Optional parameter setter

            :param fitness: The maximum fitness the algorithm looks for, stopping it's evolution when found,
            may be the biggest integer possible in the case that the optimum result is unknown
            :return: The current Builder instance
            """
            self.__max_fitness = fitness
            return self

        def with_survivors(self, percentage: float) -> 'GeneticGuesser.Builder':
            """
            Optional parameter setter

            :param percentage: The percentage of the individuals population that survives the selection process,
            only those with the highest fitness can be among this group
            :return: The current Builder instance
            """
            self.__survivors_percentage = percentage
            return self

        def with_mutation_chance(self, percentage: float) -> 'GeneticGuesser.Builder':
            """
            Optional parameter setter

            :param percentage: The chance of any gene in a newly generated offspring of being randomized
            from scratch, represented as a percentage float number
            :return: The current Builder instance
            """
            self.__mutation_chance_percentage = percentage
            return self

        def with_seed(self, seed: int) -> 'GeneticGuesser.Builder':
            """
            Optional parameter setter

            :param seed: The seed for the guesser's random generator, has deterministic results
            :return: The current Builder instance
            """
            self.__seed = seed
            return self

        def build(self) -> 'GeneticGuesser':
            """
            Constructs a GeneticGuesser from the passed parameters

            May raise an AttributeError in case an obligatory parameter wasn't specified before calling this method

            :raises: AttributeError
            :return: A new Genetic Guesser
            """
            if (self.__individuals_amount is not None) and (
                    self.__genes_per_individual is not None) and (
                    self.__gene_alphabet is not None):
                return GeneticGuesser(individuals_amount=self.__individuals_amount,
                                      genes_per_individual=self.__genes_per_individual,
                                      gene_alphabet=self.__gene_alphabet,
                                      evaluation_function=self.__evaluating_function,
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
    """
    Simple comparator function for lists

    If the lists have the same elements then it will return their length

    1 is subtracted from the word's length value for each element in them that's not the same

    :param word_1: First word to compare
    :param word_2: Second word to compare
    :return: The amount of characters the words have in common, taking order into account
    """
    assert (len(word_1) == len(word_2))
    result = len(word_1)
    for index in range(len(word_1)):
        if word_1[index] != word_2[index]:
            result -= 1
    return result


# Directories for saving plots
length_test_plots_saving_directory = './../../plots/word_guessing/length_test'
individuals_test_plots_saving_directory = './../../plots/word_guessing/individuals_test'
survivors_test_plots_saving_directory = './../../plots/word_guessing/survivors_test'
mutation_test_plots_saving_directory = './../../plots/word_guessing/mutation_test'
all_test_plots_saving_directory = './../../plots/word_guessing/individuals_survivors_mutation_test'
alphabet_test_plots_saving_directory = './../../plots/word_guessing/alphabet_test'

# Full english alphabet
english_alphabet = list(string.ascii_lowercase + string.ascii_uppercase)


def main(word_length: int, alphabet: list, individuals_amount: int, survivors_percentage: float,
         mutation_chance_percentage: float, seed: int = None, save_directory: str = None, print_info: bool = True) \
        -> int:
    """
    Main function for testing if the genetic algorithm above works as intended

    Generates a random word from the specified alphabet and then iterates the algorithm until it has guessed it
    perfectly

    A GeneticGuesser is constructed with the parameters specified above

    The Genetic guesser is passed the function word_comparator with the target word as an argument, to be used as it's
    fitness evaluation function

    A seed may be specified

    Will generate a plot with the maximum fitness found each generation, this plot can be saved into memory if a
    directory is specified for it, otherwise it will pop into a plot window

    Can print information about what it's doing if needed

    :param word_length: Length of the word to guess
    :param alphabet: Alphabet from which the word is constructed
    :param individuals_amount: Amount of individuals the GeneticGuesser maintains
    :param survivors_percentage: Percentage of selection survivors of the Guesser's algorithm
    :param mutation_chance_percentage: Percentage chance of a gene to mutate within the Guesser
    :param seed: Seed for the experiment
    :param save_directory: Directory for saving the result's plot
    :param print_info: Whether information on the guessing process should be printed
    :return: The amount of iterations needed to solve the problem
    """

    # The seed is set
    a_random = random.Random()
    if seed is not None:
        a_random.seed(seed + 12345)

    # Some information on the parameters passed is printed
    if print_info:
        first_10_alphabet = str(alphabet[0])
        for i in range(min(len(alphabet) - 1, 9)):
            first_10_alphabet += "," + str(alphabet[i + 1])
        first_10_alphabet += "...[" + str(len(alphabet) - 10) + " more elements]" if len(alphabet) > 10 else ""
        print("\nWord of length " + str(word_length) + "\nAnd alphabet: \n    " + first_10_alphabet)

    # Values are initialized
    iteration_counter = 0
    max_fitness_scores = []

    # The random word is generated
    target_word = []
    for _ in range(word_length):
        target_word.append(alphabet[a_random.randint(0, len(alphabet) - 1)])

    # The word's first 10 elements are printed
    if print_info:
        first_10_word = str(target_word[0])
        for i in range(min(len(target_word) - 1, 9)):
            first_10_word += str(target_word[i + 1])
        first_10_word += "..." if len(target_word) > 10 else ""
        print("\nThe target word is:\n    " + str(first_10_word))

    # Since the guesser can technically get stuck on local optima, after a certain amount of unfruitful iterations
    # it's reset within this loop
    while True:

        initial_iteration_counter = iteration_counter

        guesser = GeneticGuesser.Builder() \
            .with_individuals(individuals_amount) \
            .with_genes_amount(word_length) \
            .with_alphabet(alphabet) \
            .with_evaluation_function(lambda gene: word_comparator(gene, target_word)) \
            .with_max_fitness(word_length) \
            .with_survivors(survivors_percentage) \
            .with_mutation_chance(mutation_chance_percentage) \
            .with_seed(seed) \
            .build()

        # Guessing loop, will stop automatically after a certain amount of unfruitful iterations
        while iteration_counter < initial_iteration_counter + int(0.1*(word_length * len(alphabet))**2) and \
                not guesser.generational_step():
            max_fitness_scores.append(guesser.get_max_fitness())
            iteration_counter += 1
        max_fitness_scores.append(guesser.get_max_fitness())
        iteration_counter += 1

        # In case the result was found
        if iteration_counter != initial_iteration_counter + (word_length * len(alphabet) * 10) + 1:
            break
        # If not, the guesser is reset with a different seed
        else:
            seed += 1

    # The results are plotted
    _, ax = plt.subplots()

    ax.plot(range(iteration_counter), max_fitness_scores)
    ax.set_xlim([0, iteration_counter])
    ax.set_ylim([max(min(min(max_fitness_scores), word_length - 100), 0), word_length])

    # Some information is added
    plt.title("\nGenerational fitness for a genetic word guesser of length '" + str(word_length) +
              "' \namount of gene individuals '" + str(individuals_amount) +
              "' \npercentage of survivors per generation '" + str(survivors_percentage) +
              "' \nand mutation chance per gene '" + str(mutation_chance_percentage) + "'")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness")
    ax.grid(True)

    # The plot is either saved or shown
    if save_directory is not None:
        plt.savefig(save_directory + "/plot_word" + str(word_length) + "_individuals" +
                    str(individuals_amount) + "_survivors" + str(survivors_percentage) + "_mutation" +
                    str(mutation_chance_percentage) + ".png", bbox_inches='tight')
    else:
        plt.show()

    # Information on whether the solution was found is printed
    if print_info:
        print("\nThe solution was found after " + str(iteration_counter) + " iterations of the guesser")

    plt.close()

    return iteration_counter


def plot_result(x: list, y: list, title: str, xlab: str, ylab: str, save_directory: str = None) -> None:
    """
    Function for generating final comparison plots

    May save them if a directory is specified

    :param x: List of x values
    :param y: List of y values
    :param title: Plot title
    :param xlab: Plot x axis label
    :param ylab: Plot y axis label
    :param save_directory: Directory for saving the plot
    """
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
    """
    Execution of the file
    
    Tests the parameters of a GeneticGuesser instance to see what values work best
    
    Saves the resulting plots in the plots/word_guessing directory
    """

    points = 20

    binary_alphabet = ['0', '1']

    s = 1

    #   ------  Finding Length  ------
    """
    Test for comparing the amount of iterations needed when word guessing for different length of a binary word
    """

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
    """
    Test for seeing the effects the amount of individuals parameter has on the amount of iterations
    """

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
    """
    Test for seeing the effects the amount of survivors parameter has on the amount of iterations
    """

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
    """
    Test for seeing the effects the mutation chance parameter has on the amount of iterations
    """

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
    """
    Deprecated test for seeing the effects of all of these parameters at once
    
    Simply takes too long to execute
    """

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
    """
    Test for seeing the amount of iterations needed to guess a word using the full english alphabet, 
    for different word lengths
    """

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
