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
        for i in range(int(len(self.__genes)*(1.0-self.__survivors_percentage/100.0))):
            index = self.__fitness_scores.index(min(self.__fitness_scores))
            self.__genes.pop(index)
            self.__fitness_scores.pop(index)
        return

    def __reproduce(self) -> None:
        new_genes = []
        for i in range(self.__individuals_amount):
            index_1 = self.__random.randint(0, len(self.__genes)-1)
            index_2 = self.__random.randint(0, len(self.__genes)-1)
            if len(self.__genes) != 2:
                while index_1 == index_2:
                    index_2 = self.__random.randint(0, len(self.__genes)-1)
            elif index_1 == index_2:
                index_2 = 0 if index_1 == 1 else 1
            gene = []
            for j in range(self.__genes_per_individual):
                r = self.__random.uniform(0.0, 100.0+self.__mutation_chance_percentage)
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
                return GeneticGuesser(self.__individuals_amount,
                                      self.__genes_per_individual,
                                      self.__gene_alphabet,
                                      self.__evaluating_function,
                                      self.__max_fitness,
                                      self.__survivors_percentage,
                                      self.__mutation_chance_percentage,
                                      self.__seed)

            error_message = ""
            if self.__individuals_amount is None:
                error_message += "Amount of Individuals was not specified\n"
            if self.__genes_per_individual is None:
                error_message += "Amount of Genes per Individual was not specified\n"
            if self.__gene_alphabet is None:
                error_message += "Evaluator Function for Individuals was not specified\n"
            raise AttributeError(error_message+"GeneticGuesser Build Failed")


def word_comparator(word_1: list, word_2: list) -> int:
    assert(len(word_1) == len(word_2))
    diff = len(word_1)
    for i in range(len(word_1)):
        if word_1[i] == word_2[i]:
            diff -= 1
    return diff


def main(word_length: int = 5, alphabet=None, seed: int = 1234567):

    if alphabet is None:
        alphabet = ["0", "1"]

    iterations = 0
    max_fitness_scores = []

    guesser = GeneticGuesser.Builder()\
        .with_alphabet(alphabet)\
        .with_individuals(word_length*3)\
        .with_genes_amount(word_length)\
        .with_evaluating_function(lambda gene: word_comparator(gene, word))\
        .with_survivors(25.0)\
        .with_mutation_chance(5.0)\
        .with_seed(seed)\
        .build()

    # A random number generator is constructed, with a different seed made
    # from the original one
    a_random = random.Random()
    a_random.seed(seed+1)
    # Otherwise both the guesser and main() generate the same first word

    word = []
    for i in range(word_length):
        word.append(alphabet[a_random.randint(0, len(alphabet)-1)])

    while not guesser.generational_step():
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1
    max_fitness_scores.append(guesser.get_max_fitness())
    iterations += 1

    _, ax = plt.subplots()

    ax.plot(range(iterations), max_fitness_scores, 'o')
    ax.set_xlim([0, iterations])
    ax.set_ylim([0, word_length])

    plt.title("Fitness of each generation for a genetic guesser of length " +
              str(word_length))
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main(word_length=300)
