import random

import matplotlib.pyplot as plt


class GeneticGuesser:

    def __init__(self, gene_amount: int, word_length: int, alphabet: list,
                 seed: int = None, survivors_amount: int = 0):
        self.__gene_amount = gene_amount
        self.__gene_length = word_length
        self.__alphabet = alphabet
        self.__genes = []
        self.__fitness_scores = []
        self.__survivors_amount = survivors_amount

        self.__evaluator_function = self.__null_evaluation_function
        self.__max_fitness = self.__gene_length

        self.__random = random.Random()
        if seed is not None:
            self.__random.seed(seed)

        for i in range(gene_amount):
            gene = []
            for j in range(word_length):
                gene.append(alphabet[self.__random.randint(0, len(alphabet)-1)])
            self.__genes.append(gene)
        pass

    def set_evaluation_function(self, fn) -> None:
        self.__evaluator_function = fn
        pass

    def set_max_possible_fitness(self, fitness: int) -> None:
        self.__max_fitness = fitness

    @staticmethod
    def __null_evaluation_function(word: list) -> int:
        return len(word)

    def __evaluate(self) -> None:
        self.__fitness_scores = []
        for gene in self.__genes:
            fitness = self.__evaluator_function(gene)
            self.__fitness_scores.append(fitness)
        pass

    def is_done(self) -> bool:
        for evaluation in self.__fitness_scores:
            if evaluation == self.__max_fitness:
                return True
        return False

    def get_max_fitness(self) -> int:
        return max(self.__fitness_scores)

    def get_best_word(self) -> list:
        return self.__genes[self.__fitness_scores.index(self.get_max_fitness())]

    def __select(self) -> None:
        for i in range(len(self.__genes)-self.__survivors_amount):
            index = self.__fitness_scores.index(min(self.__fitness_scores))
            self.__genes.pop(index)
            self.__fitness_scores.pop(index)
        pass

    def __reproduce(self) -> None:
        new_genes = []
        for i in range(self.__gene_amount):
            index_1 = self.__random.randint(0, len(self.__genes)-1)
            index_2 = self.__random.randint(0, len(self.__genes)-1)
            if len(self.__genes) != 2:
                while index_1 == index_2:
                    index_2 = self.__random.randint(0, len(self.__genes)-1)
            elif index_1 == index_2:
                index_2 = 0 if index_1 == 1 else 1
            gene = []
            for j in range(self.__gene_length):
                r = self.__random.randint(1, 101)
                if r <= 50:
                    gene.append(self.__genes[index_1][j])
                elif r <= 100:
                    gene.append(self.__genes[index_2][j])
                else:
                    gene.append(self.__alphabet[self.__random.randint(0, len(self.__alphabet)-1)])

            new_genes.append(gene)
        self.__genes = new_genes
        pass

    def generational_step(self) -> bool:
        if len(self.__fitness_scores) == 0:
            self.__evaluate()
        if not self.is_done():
            self.__select()
            self.__reproduce()
            self.__evaluate()
        return self.is_done()


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

    guesser = GeneticGuesser(word_length * 3, word_length, alphabet,
                             seed=seed, survivors_amount=5)

    a_random = random.Random()
    a_random.seed(seed+1)

    word = []
    for i in range(word_length):
        word.append(alphabet[a_random.randint(0, len(alphabet)-1)])

    guesser.set_evaluation_function(lambda gene: word_comparator(gene, word))

    while not guesser.generational_step():
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1
    max_fitness_scores.append(guesser.get_max_fitness())
    iterations += 1

    _, ax = plt.subplots()

    ax.plot(range(iterations), max_fitness_scores, 'o')
    ax.set_xlim([0, iterations])
    ax.set_ylim([0, word_length])

    plt.title("Fitness of each generation for a genetic guesser of length "+str(word_length))
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main(word_length=300)
