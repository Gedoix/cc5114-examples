# Neural Networks and Genetic Programming
---

Examples of the concepts from the course CC5114: Neural Networks 
and Genetic Programming

Copy hosted over at my [Github](https://github.com/Gedoix/cc5114-examples.git) page (Collaborators only)]

Packages are described here in the order they were developed, and explaining 
what they do

## Description of the Packages in the `src` Directory
---

### `basic_perceptrons`

This package contains basic implementations of perceptrons and 
networks made from them, along with a more specific collection of 
single perceptron logic gates and a bit adder network of nand gates

`basic_perceptrons.py` contains a perceptron implementation and 
all logic gate implementations, including some extra methods for 
use outside the package

`basic_networks.py` contains a basic yet extensible network 
implementation as well as the addition network

### `learning_perceptrons`

This package contains a basic implementation of a learning linear 
classifier re-using the perceptron implementation from `basic_perceptrons.py`

`basic_classifier.py` contains an implementation of a linear classifier that 
auto-trains if necessary

The file can be executed, in which case it will print example plots that prove 
that it can learn

### `sigmoid_perceptrons`

This package implements sigmoid perceptrons and a classifier that uses them, 
reusing some code from the two previous packages

`sigmoid_perceptrons.py` contains an implementation of both the sigmoid 
perceptron and a classifier using it

The file can be executed, in which case it will print example plots that prove 
that it can learn

`logic_gate_training.py` contains a plotted test for checking if the sigmoid 
perceptron can simulate the `and`, `or`, `nand` and `xor` logic gates through 
learning the correct parameters

Learning `xor` using a single perceptron is impossible, so the accuracy of 
that simulation in particular can never arrive at 100%, but the other gates 
can all be learned

### `sigmoid_networks`

This package contains an implementation from scratch of an extensible and 
trainable neural network, using sigmoid perceptrons, to be used for 
classification problems in large datasets.

`network.py` is an executable file, which generates plots to prove it can 
actually learn a simple linear classification problem with many neurons 
and also what sort of results and consistency to expect from it.

### `dataset_predictor`

This package contains a single executable file, configured to read a `.csv` 
like dataset, in this case `letter-recognition.data`, and use the contents of 
the `sigmoid_networks` package to train a complex network to predict with good 
accuracy the value of one of the dataset's attributes from the rest of it's 
attributes.

The file, `network_prediction.py`, then produces 6 plots showing the 
improvement over times trained of the network's classification algorithm 
through the use of 6 different metrics classification metrics.

### `file_utilities`

This package includes the file `dir_management.py` with code from
[this](http://stackoverflow.com/questions/1889597/deleting-directory-in-python)
stack overflow answer.

It's a simple package for storing functions that manipulate files and
directories.

In this case `dir_management.py` contains the function `clear_dir(path_)`
which deletes all files inside the directory specified by `path_`

This function is useful when reseting saved files inside of the `plots`
directory.

### `genetic_algorithms`

This package contains an implementation of a genetic guessing algorithm, in
the file `genetic_algorithm.py`.

Also, a solver for the [N-Queen Problem](Homework_2_Neural_Networks.pdf) can
be found in `n_queen_optimizer.py`.

This last file contains some code from 
[this](https://stackoverflow.com/a/10195347/10216044) stack overflow answer,
regarding how to plot a chess board using the matplotlib package.

Both of these files are executable.

### `neuroevolution_algorithms`

This package contains an implementation of the N.E.A.T. algorithm as 
described in [this](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) 
paper.

The classes the algorithm uses are inside the file `neat_network.py`, and the 
main algorithm in `NEAT.py`

### `snake`

This package contains both an implementation of the classic Snake game, 
executable in the file `snake_game.py` for testing, and a test of the N.E.A.T. 
algorithm inside the Snake game's environment.

## Description of other Non-Source Files
---

### The `venv` Directory

Contains the Python 3.6 virtual environment of the project.

Pycharm can manage it, there's really no reason to look into it.

### The `test` Directory

Houses Unit tests for some classes from `src`.

These files can be executed with instructions below.

### The `resources` Directory

Contains a downloaded copy of 
[this](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) 
dataset, analyzed when testing the `dataset_predictor` package.

### The `plots` Directory

Contains `.png` saves of plots generated from executable files in `src` and some 
`.pdf` files with execution prints.

May also contain `.gif` and `.txt` saved results.

### `PDF` Documents

The files:

* `Homework_1_Neural_Networks.pdf` [here](Homework_1_Neural_Networks.pdf)

* `Homework_2_Neural_Networks.pdf` [here](Homework_2_Neural_Networks.pdf)

* `Homework_3_Neural_Networks.pdf` [here](Homework_3_Neural_Networks.pdf)

Contain further documentation and analysis on the algorithms and experiments 
implemented in the project.

## Getting Started
---

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes.

### Prerequisites

This repo's project was and is being developed using 
[Python 36](https://www.python.org/downloads/), together 
with the packages [numpy](http://www.numpy.org/), 
[matplotlib](https://matplotlib.org/), 
[tqdm](https://github.com/tqdm/tqdm) and [PyGame](https://www.pygame.org/news).

All of this was put together easily through [Jetbrain's](https://www.jetbrains.com/)
IDE for Python, [PyCharm](https://www.jetbrains.com/pycharm/)

It is recommended that this same IDE is used for testing and evaluating the 
project, the explanations on how to run it will not include any other 
environments.

It will also be assumed that the reader's OS of choice is a Linux Debian distro 

### Installing

The easiest ways to install all of these dependencies is as follows:

* Getting a copy of [Jetbrain's Toolbox](https://www.jetbrains.com/products.html?fromMenu#) 
and installing it, once the tool is running it should allow installing either 
Pycharm Community Edition or the Professional Edition, either one is useful

* Getting [git](https://git-scm.com/) installed in the machine, this will 
allow Jetbrain's tools to recognize it's presence automatically, and 
facilitate project version control

* Cloning the repository from my 
[Github](https://github.com/Gedoix/cc5114-examples.git) page (Collaborators only)
using the IDE's facilitation tools for Version Control Importing. Make sure to
specify that the project was developed in PyCharm and not any other IDE through
the import's UI

* Once the repo has been cloned, installing Python 3.6 is easy following 
[this](http://unix.stackexchange.com/questions/110014/ddg#110163) 
tutorial

* After all the already mentioned is installed, the project needs to be opened,
 so PyCharm can configure the interpreter to Python 3.6
 
* Accessing the project settings inside the toolbar, going into the 
`interpreter` tab, and clicking on the option `project interpreter`, `show all...`
will allow the project to set the interpreter included with the code.

* In this menu, simply press the `plus` icon, select `existing environment`, 
and set the desired directory to the project's `venv/bin/python` directory. 

* After this the projects imports should all be working. Time to test it out!

## Running the tests

To run the automated unit tests all that needs to be done is 
to right-click the `test` directory and select the `Run 'Unittest in test' 
with Coverage` option, a coverage suite should open along with a console 
stating `Tests passed: 22 of 22 tests`.

This automated tests only offer basic coverage of some of the more advanced 
methods in the project, to run some of the more lengthy (and time consuming) 
tests all that's needed is to right-click the Python executable files 
containing them and click `Run 'file_to_execute.py'`. The files that can be 
run are:

* `learning_perceptrons.basic_classifier.py` [here](src/learning_perceptrons/basic_classifier.py)

* `sigmoid_perceptrons.sigmoid_perceptrons.py` [here](src/sigmoid_perceptrons/sigmoid_perceptrons.py)

* `sigmoid_perceptrons.logic_gate_training.py` [here](src/sigmoid_perceptrons/logic_gate_training.py)

* `sigmoid_networks.network.py` [here](src/sigmoid_networks/network.py)

* `dataset_predictor.network_prediction.py` [here](src/dataset_predictor/network_prediction.py)

* `genetic_algorithms.genetic_algorithm.py` [here](src/genetic_algorithms/genetic_algorithm.py)

* `genetic_algorithm.n_queen_optimizer.py` [here](src/genetic_algorithms/n_queen_optimizer.py)

* `snake.snake_game.py` [here](src/snake/snake_game.py)

* `snake.NEAT_experiment.py` [here](src/snake/NEAT_experiment.py)

All of these files can generate plots and printed proof of the workings
of the code being tested

The execution of `genetic_algorithms.genetic_algorithm.py` and 
`genetic_algorithm.n_queen_optimizer.py` in particular saves the 
generated plots to the `plots` directory, overwriting any previous saves

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - IDE used for Python
* [git](https://git-scm.com/) - Version Control system
* [NumPy](http://www.numpy.org/) - Scientific numeric computation package
* [MatPlotLib](https://matplotlib.org/) - Plotting package
* [tqdm](https://github.com/tqdm/tqdm) - Loading bar printer and manager
* [PyGame](https://www.pygame.org/news) - Snake game simulation
* [README.md Template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) - Template for good practices when writing a README

## Versioning

I used [GitHub](http://github.com/) for versioning.

## Authors

* **Diego Ortego** - *All Package Implementations* - [Gedoix](https://github.com/Gedoix)

## Acknowledgments

* Hat tip to anyone whose code was used
* Universidad de Chile's CC5114 course
