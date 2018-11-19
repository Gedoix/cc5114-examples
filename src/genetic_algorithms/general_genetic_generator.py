import random

import matplotlib.pyplot as plt


class GeneticWordGuesser:

    def __init__(self, gene_amount: int, word_length: int, alphabet: list,
                 seed: int = None, survivors_amount: int = 0):
        self.__gene_amount = gene_amount
        self.__gene_length = word_length
        self.__alphabet = alphabet
        self.__genes = []
        self.__fitness_scores = []
        self.__survivors_amount = survivors_amount

        self.__random = random.Random()
        if seed is not None:
            self.__random.seed(seed)

        for i in range(gene_amount):
            gene = []
            for j in range(word_length):
                gene.append(alphabet[self.__random.randint(0, len(alphabet)-1)])
            self.__genes.append(gene)
        pass

    def __evaluate(self, expected_word: list):
        self.__fitness_scores = []
        for i in range(self.__gene_amount):
            fitness = self.__gene_length
            for j in range(self.__gene_length):
                if expected_word[j] != self.__genes[i][j]:
                    fitness -= 1
            self.__fitness_scores.append(fitness)
        pass

    def is_done(self) -> bool:
        for evaluation in self.__fitness_scores:
            if evaluation == self.__gene_length:
                return True
        return False

    def get_max_fitness(self) -> int:
        return max(self.__fitness_scores)

    def __select(self):
        for i in range(len(self.__genes)-self.__survivors_amount):
            index = self.__fitness_scores.index(min(self.__fitness_scores))
            self.__genes.pop(index)
            self.__fitness_scores.pop(index)
        pass

    def __reproduce(self):
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
                r = self.__random.randint(1, 110)
                if r <= 50:
                    gene.append(self.__genes[index_1][j])
                elif r <= 100:
                    gene.append(self.__genes[index_2][j])
                else:
                    gene.append(self.__alphabet[self.__random.randint(0, len(self.__alphabet)-1)])

            new_genes.append(gene)
        self.__genes = new_genes

    def generational_step(self, word: list) -> bool:
        if len(self.__fitness_scores) == 0:
            self.__evaluate(word)
        if not self.is_done():
            self.__select()
            self.__reproduce()
            self.__evaluate(word)
        return self.is_done()


def main(word_length: int = 5, alphabet=None, seed: int = 1234567):

    if alphabet is None:
        alphabet = ["0", "1"]

    iterations = 0
    max_fitness_scores = []

    guesser = GeneticWordGuesser(word_length*3, word_length, alphabet,
                                 seed=seed, survivors_amount=5)

    a_random = random.Random()
    a_random.seed(seed+1)

    word = []
    for i in range(word_length):
        word.append(alphabet[a_random.randint(0, len(alphabet)-1)])

    while not guesser.generational_step(word):
        print(guesser.get_max_fitness())
        max_fitness_scores.append(guesser.get_max_fitness())
        iterations += 1
    max_fitness_scores.append(guesser.get_max_fitness())
    iterations += 1

    _, ax = plt.subplots()

    ax.plot(range(iterations), max_fitness_scores, 'o')
    ax.set_xlim([0, iterations])
    ax.set_ylim([0, word_length])

    plt.title("Maximum Fitness of each generation for a genetic guesser of a word of length"+str(word_length))
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main(word_length=1000)
