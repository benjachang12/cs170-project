import networkx as nx
import time
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os

"""
CS170 Spring 2020 Project

Kelvin Pang, Ben Chang

"""

"""
Algorithm:

Maxmimum Leaf Implementation

- Get most leaves
- Delete leaves
- Have shortest tree with a set that approximately dominates graph

"""

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    T = nx.minimum_spanning_tree(G)
    X = list(T.nodes)
    length = len(X)
    U = T
    for i in range(length):
        a = X[i]
        if T.degree[a] == 1:
            U.remove_node(a)
    T = U
    return T


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == "__main__":
    output_dir = "experiment_outputs/test2"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")

        # Solve problem and time the elapsed time
        start_time = time.time()
        T = solve(G)
        elapsed_time = time.time() - start_time

        print("Finished solving:", graph_name)
        print('With total runtime:', elapsed_time, "(s)")
        print("With average  pairwise distance: {}".format(average_pairwise_distance(T)), "\n")

        write_output_file(T, f"{output_dir}/{graph_name}.out")
