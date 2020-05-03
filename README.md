# cs170-project

CS170 Spring 2020 Project

Project spec can be found in spec.pdf

Requirements:
- networkx: https://networkx.github.io/documentation/stable/install.html

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: contains the project algorithm implementation
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

## Project Run Instructions:
Ensure that `solver.py`, \inputs, and \phase2_outputs are all in the root directory
- \inputs should contains all of the input files
- \phase2_outputs should be an empty directory, otherwise contents will be overwritten

### Running the solver on a single graph:
This runs the solver on a single graph, producing plots of the hyperparameter tuning, visualizing the graphs, and outputting the different costs and runtime.

Modify main function in `solver.py`:
- Set `test_single_graph` = True
- Set `path` = input graph filepath (e.g. path = "inputs\\medium-9.in")

Run: python solver.py

### Running the solver on a all inputs:
This runs the solver on all input graph, outputting the cost of each graph, total runtime, and the average cost across all graphs.

Modify main function in `solver.py`:
- Set `test_single_graph` = False
- Set `generating_outputs` = True (if you want to write outputs to output directory)
- Set `generating_outputs` = False (if you don't want to save outputs)

## Submission Workflow
Run: python prepare_submission.py phase2_outputs/ submission.json

