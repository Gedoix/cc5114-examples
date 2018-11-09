# cc5114-examples
---

Examples of the concepts from the course CC5114: Neural Networks 
and Genetic Programming

Copy hosted over at [my Github page (Collaborators only)](https://github.com/Gedoix/cc5114-examples.git)

Packages are described here in the order they were developed, and explaining 
what they do

## Description of the Packages
---

### basic_perceptrons

This package contains basic implementations of perceptrons and 
networks made from them, along with a more specific collection of 
single perceptron logic gates and a bit adder network of nand gates

`basic_perceptrons.py` contains a perceptron implementation and 
all logic gate implementations, including some extra methods for 
use outside the package

`basic_networks.py` contains a basic yet extensible network 
implementation as well as the addition network

### learning_perceptrons

This package contains a basic implementation of a learning linear 
classifier re-using the perceptron implementation from `basic_perceptrons.py`

`basic_classifier.py` contains an implementation of a linear classifier that 
auto-trains if necessary

The file can be executed, in which case it will print example plots that prove 
that it can learn

### sigmoid_perceptrons

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

### sigmoid_networks

This package contains an implementation from scratch of an extensible and 
trainable neural network, using sigmoid perceptrons, to be used for 
classification problems in large datasets.

`network.py` is an executable file, which generates plots to prove it can 
actually learn a simple linear classification problem with many neurons 
and also what sort of results and consistency to expect from it.

### dataset_predictor

This package contains a single executable file, configured to read a `.csv` 
like dataset, in this case `letter-recognition.data`, and use the contents of 
the `sigmoid_networks` package to train a complex network to predict with good 
accuracy the value of one of the dataset's attributes from the rest of it's 
attributes.

The file, `network_prediction.py`, then produces 6 plots showing the 
improvement over times trained of the network's classification algorithm 
through the use of 6 different metrics classification metrics.

## Getting Started
---

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes.

### Prerequisites

This repo's project was and is being developed using [Python 3.7](), together 
with the packages [numpy](http://www.numpy.org/), 
[matplotlib](https://matplotlib.org/) and [tqdm](https://github.com/tqdm/tqdm).

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

* Cloning the repository from [my Github page (Collaborators only)](https://github.com/Gedoix/cc5114-examples.git)
using the IDE's facilitation tools for Version Control Importing. Make sure to
specify that the project was developed in PyCharm and not any other IDE through
the import's UI

* Once the repo has been cloned, installing Python 3.7 is easy following [this](http://unix.stackexchange.com/questions/110014/ddg#110163) 
tutorial

* After all the already mentioned is installed, the project needs to be opened,
 so PyCharm can configure the interpreter to Python 3.7
 
* Accessing the project settings inside the toolbar going into the 
`interpreter` tab, and clicking on the plus `add package` button will allow to 
easily install [numpy](http://www.numpy.org/), 
[matplotlib](https://matplotlib.org/) and [tqdm](https://github.com/tqdm/tqdm),
just by searching them by name.

* After this the projects imports should all be working. Time to test it out!

## Running the tests

Running the tests

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc