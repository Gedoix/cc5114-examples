# cc5114-examples
---

Examples of the concepts from the course CC5114: Neural Networks 
and Genetic Programming

Packages are numbered in the order they were developed

## basic_perceptrons
---

This package contains basic implementations of perceptrons and 
networks made from them, along with a more specific collection of 
single perceptron logic gates and a bit adder network of nand gates

`basic_perceptrons.py` contains a perceptron implementation and 
all logic gate implementations, including some extra methods for 
use outside the package

`basic_networks.py` contains a basic yet extensible network 
implementation as well as the addition network

## learning_perceptrons
---

This package contains a basic implementation of a learning linear 
classifier re-using the perceptron implementation from `basic_perceptrons.py`

`basic_classifier.py` contains an implementation of a linear classifier that 
auto-trains if necessary

The file can be executed, in which case it will print example plots that prove 
that it can learn

## sigmoid_perceptrons
---

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

## sigmoid_networks
---


