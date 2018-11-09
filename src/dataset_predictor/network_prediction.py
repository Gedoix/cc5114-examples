import csv

import numpy as np

import matplotlib.pyplot as plt

import tqdm

from sigmoid_networks.network import Network

ALPHABET = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
            'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25}


def data_getter(disable_bar: bool = False) -> (np.ndarray, np.ndarray):
    pre_classes = []
    pre_attributes = []

    with open("./../../resources/letter-recognition.data", "r") as file:

        reader = csv.reader(file, delimiter=',')

        for row in tqdm.tqdm(reader, desc="Loading File", unit=" data examples", disable=disable_bar):
            c = []
            for i in range(26):
                c.append(0.0)
                if i == ALPHABET.get(row[0]):
                    c[i] = 1.0
            pre_classes.append(np.array(c))
            attribute = []
            for i in range(len(row) - 1):
                attribute.append(float(row[i + 1]))
            pre_attributes.append(np.array(attribute))

        classes = np.zeros((len(pre_classes[0]), len(pre_classes)))
        attributes = np.zeros((len(pre_attributes[0]), len(pre_attributes)))

        for i in tqdm.tqdm(range(len(pre_classes)), desc="Formatting Data", unit=" data examples", disable=disable_bar):
            for j in range(len(pre_classes[0])):
                classes[j, i] = pre_classes[i][j]
            for j in range(len(pre_attributes[0])):
                attributes[j, i] = pre_attributes[i][j]

        return attributes, classes


def data_normalizer(attributes: np.ndarray, disable_bar: bool = False) -> np.ndarray:
    maximums = []
    minimums = []
    for i in range(16):
        maximums.append(0.0)
        minimums.append(0.0)

    maximum = np.amax(attributes, 1)
    minimum = np.amin(attributes, 1)

    for i in tqdm.tqdm(range(len(attributes[0, :])), desc="Normalizing Attributes", unit=" attributes",
                       disable=disable_bar):
        attributes[:, i] = np.divide(np.subtract(attributes[:, i], minimum), np.subtract(maximum, minimum))

    return attributes


def data_partitioner(attributes: np.ndarray, classes: np.ndarray, proportion: float = 0.1, seed: int = None) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    length = len(attributes[0, :])
    if seed is not None:
        np.random.seed(seed=seed)
    np.random.shuffle(attributes)
    np.random.shuffle(classes)
    training_attributes = attributes[:, range(int(length * (1.0 - proportion)))]
    testing_attributes = attributes[:, range(int(length * (1.0 - proportion)), length)]
    training_classes = classes[:, range(int(length * (1.0 - proportion)))]
    testing_classes = classes[:, range(int(length * (1.0 - proportion)), length)]

    return np.array(training_attributes), np.array(testing_attributes), \
        np.array(training_classes), np.array(testing_classes)


def main(epochs: int = 500, seed: int = None, disable_bar: bool = False):
    attributes, classes = data_getter(disable_bar=disable_bar)
    attributes = data_normalizer(attributes, disable_bar=disable_bar)
    train_inputs, test_inputs, train_outputs, test_outputs = data_partitioner(attributes, classes, 0.1, seed=seed)

    network = Network(len(train_inputs[:, 0]), [18, 20, 22, 24], 26)

    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "Specificity": [], "Mean error": [],
               "Mean squared error": []}

    print("Initial prediction capacity")

    network.generate_metrics_epoch(test_inputs, test_outputs)

    metrics.get("Accuracy").append(network.get_accuracy()*100)
    metrics.get("Precision").append(network.get_precision()*100)
    metrics.get("Recall").append(network.get_recall()*100)
    metrics.get("Specificity").append(network.get_specificity()*100)
    metrics.get("Mean error").append(network.get_mean_error()*100)
    metrics.get("Mean squared error").append(network.get_mean_squared_error()*100)

    for _ in tqdm.tqdm(range(epochs), desc="Training network", unit=" epochs", disable=disable_bar):
        network.train_epoch(train_inputs, train_outputs)
        network.generate_metrics_epoch(test_inputs, test_outputs)

        metrics.get("Accuracy").append(network.get_accuracy()*100)
        metrics.get("Precision").append(network.get_precision()*100)
        metrics.get("Recall").append(network.get_recall()*100)
        metrics.get("Specificity").append(network.get_specificity()*100)
        metrics.get("Mean error").append(network.get_mean_error()*100)
        metrics.get("Mean squared error").append(network.get_mean_squared_error()*100)

    for key in metrics.keys():

        fig, ax = plt.subplots()
        ax.plot(range(epochs + 1), metrics.get(key), 'o')
        ax.set_xlim([0, epochs])
        if key not in ["Mean error", "Mean squared error"]:
            ax.set_ylim([0, 100])

        plt.title(key + " of classifier v/s epochs trained for a total of " + str(epochs) + " epochs")
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    main(seed=1907572537)
