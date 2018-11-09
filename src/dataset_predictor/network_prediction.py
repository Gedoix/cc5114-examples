import csv

import numpy as np

import tqdm as tqdm

ALPHABET = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
            'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25}


def main():
    with open("./../../resources/letter-recognition.data", "r") as file:
        print("Loading Files\n")
        reader = csv.reader(file, delimiter=',')
        classes = []
        attributes = []
        bar = tqdm.tqdm()
        for row in reader:
            c = []
            for i in range(26):
                c.append(0.0)
                if i == ALPHABET.get(row[0]):
                    c[i] = 1.0
            classes.append(np.array(c))
            attribute = []
            for i in range(len(row) - 1):
                attribute.append(float(row[i + 1]))
            attributes.append(np.array(attribute))
            bar.update()
        bar.close()

        maximums = []
        minimums = []
        for i in range(16):
            maximums.append(0.0)
            minimums.append(0.0)

        print("Normalizing Attributes\n")
        bar = tqdm.tqdm(total=len(attributes))
        for attribute in attributes:
            for i in range(len(attribute)):
                maximums[i] = np.maximum(maximums[i], attribute[i])
                minimums[i] = np.minimum(minimums[i], attribute[i])
        for attribute in attributes:
            for i in range(len(attribute)):
                attribute[i] = float(attribute[i]-minimums[i])/float(maximums[i]-minimums[i])


if __name__ == '__main__':
    main()
