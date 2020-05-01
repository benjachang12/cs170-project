import networkx as nx
import time
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os
import matplotlib.pyplot as plt

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
    M = nx.minimum_spanning_tree(G)
    X = list(M.nodes)
    length = len(X)
    U = M.copy()
    if length > 2:
        for i in range(length):
            a = X[i]
            if U.degree[a] < 2:
                M.remove_node(a)
    T = M
    # visualize_results(G,T)
    return T

def visualize_results(G, T):
    """
    Visualizes input graph and output tree side by side for easy comparison

    :param G: input Graph
    :param T: output Tree
    :return:
    """

    plt.subplot(121)
    nx.draw(G, with_labels=True)
    plt.subplot(122)
    nx.draw(T, with_labels=True)

    plt.subplot(121).set_title('Graph')
    plt.subplot(122).set_title('Tree')
    plt.show()


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

    # input_dir = "inputs"
    # input_path = "medium-111.in"
    #
    #
    # G = read_input_file(f"{input_dir}/{input_path}")
    # T = solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')
