import networkx as nx
import numpy as np
import time
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import os



"""
CS170 Sp2020 Project: Cell Tower Network Design for Horizon Wireless


authors: Benjamin Chang, Kelvin Pang 
date: 4/29/2020
"""

def solve(G):
    """
    Solves the problem statement

    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    # build a spanning tree of n-vertex weighted graph
    # TODO: implement algorithm from the paper


    T = nx.maximum_spanning_tree(G)
    return T


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    # G = read_input_file(path)
    # T = solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')

    output_dir = "experiment_outputs/test1"
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

        # TODO: write a save_to_csv function that saves a table of inputs and their runtime
