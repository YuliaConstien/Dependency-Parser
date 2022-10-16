# Introduction

In this project the goal is to implement a neural-network based dependency parser with
the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

# Dependency parsing
The implementation will be a transition-based parser, which incrementally
builds up a parse one step at a time.
At every step it maintains a partial parse, which is represented as follows:

• A stack of words that are currently being processed.

• A buffer of words yet to be processed.

• A list of dependencies predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains
all words of the sentence in order. At each step, the parser applies a transition to the partial
parse until its buffer is empty and the stack size is 1.

The following transitions can be applied:

• SHIFT: removes the first word from the buffer and pushes it onto the stack.

• LEFT-ARC: marks the second (second most recently added) item on the stack as a
dependent of the first item and removes the second item from the stack, adding a
first_word → second_word dependency to the dependency list.

• RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of
the second item and removes the first item from the stack, adding a second_word →
first_word dependency to the dependency list.
On each step, your parser will decide among the three transitions using a neural network
classifier.

# Structure: 

the __init__ and parse_step functions in the PartialParse class in
parser_transitions.py implements the transition mechanics the parser will use.

The model extracts a feature vector representing the current state. The function extracting these features has been
implemented for you in utils/parser_utils.py.

This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the
buffer, dependent of the second-to-last word in the stack if there is one, etc.) Then the network looks up an embedding for each word at the according index and
concatenates them along a single dimension into a (flat) input vector.


In parser_model.py is the the implementation of the neural network using
PyTorch.

Execute python run.py to train the model and compute predictions on test data from
the Penn Treebank corpus (annotated with Universal Dependencies).
https://www.aclweb.org/anthology/D14-1082.pdf



