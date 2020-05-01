import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
    :param G: networkx.Graph
    :return: T networkx.Graph
    """
    # TODO: your code here!
    # build a spanning tree of n-vertex weighted graph
    # TODO: implement algorithm from the paper
    F = maximally_leafy_forest(G)


    # T = nx.maximum_spanning_tree(G)
    return T

def brute_force_search(G):
    """

    :param G:
    :return:
    """

def maximally_leafy_forest(G):
    """
    Computes the maximally leafy forest F for G

    :param G:
    :return: F networkx.Graph
    """
    F = nx.Graph()  # empty graph
    F.add_nodes_from(G.nodes)

    S = nx.utils.UnionFind()
    d = np.zeros(len(G.nodes))
    for v in G.nodes:
        S[v]
    for v in G.nodes:
        S_prime = {}
        d_prime = 0
        for u, weights in G.adj[v].items():
            if S[u] != S[v] and S[u] not in S_prime.values():
                d_prime = d_prime + 1
                # S_prime[S[u]]
                S_prime[u] = S[u]
        if d[v] + d_prime >= 3:
            for u, weights in S_prime.items():
                F.add_edge(u, v)
                S.union(S[v], S[u])
                d[u] = d[u] + 1
                d[v] = d[v] + 1
    nx.draw(G, with_labels=True)
    nx.draw(F, with_labels=True)
    return F


def connect_disjoint_subtrees(F):
    """
    Add edges to F to make it a spanning tree T of G
    :param F:
    :return:
    """


    pass

def prune_leaves(T):
    """
    Prunes the leaves of tree T ONLY if it reduces the average pairwise distance
    :param T:
    :return:
    """
    pass

def k_star(G):
    """

    :param G:
    :return:
    """
    pass

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]

    # path = "inputs\small-249.in"
    path = "inputs\small-250.in"

    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test.out')

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
