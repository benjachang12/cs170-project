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

def solve(G, visualize=False, verbose=False):
    """
    Solves the problem statement
    :param G: networkx.Graph
    :return: T networkx.Graph
    """
    # TODO: your code here!
    # build a spanning tree of n-vertex weighted graph

    F, S = maximally_leafy_forest(G)
    T = connect_disjoint_subtrees(G, F, S)
    T_pruned = prune_leaves(T)

    # visualize and compare results
    if visualize:
        visualize_results(G, F, T, T_pruned)
    if verbose:
        MST = nx.maximum_spanning_tree(G)
        print("Cost of G:", average_pairwise_distance(G))
        print("Cost of an MST:", average_pairwise_distance(MST))
        print("Cost of T:", average_pairwise_distance(T))
        print("Cost of T_pruned:", average_pairwise_distance(T_pruned))

    return T_pruned

def brute_force_search(G):
    """

    :param G:
    :return:
    """

def maximally_leafy_forest(G):
    """
    Computes the maximally leafy forest F for G
    A maximally leafy forest F is a set of disjointly "leafy" subtrees of G,
    where F is not a subgraph of any other leafy forest of G
    :param G:
    :return: F networkx.Graph
             S disjoint subtrees
    """
    F = nx.Graph()  # empty graph
    F.add_nodes_from(G.nodes)

    S = nx.utils.UnionFind()
    d = np.zeros(len(G.nodes))
    for v in G.nodes:
        S[v]


    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # for v in G.nodes:
    for v, deg in sorted_nodes:
        S_prime = {}
        d_prime = 0
        #for u, weights in G.adj[v].items():
        neighbors = list(G.neighbors(v))
        sorted_neighbors = sorted(neighbors, key=lambda x: G.edges[(x,v)]['weight'])
        for u in neighbors:
            if S[u] != S[v] and S[u] not in S_prime.values():
                d_prime = d_prime + 1
                S_prime[u] = S[u]
        if d[v] + d_prime >= 3:
            # for u, weights in S_prime.items():
            for u in S_prime.keys():
                F.add_edge(u, v)
                S.union(S[v], S[u])
                d[u] = d[u] + 1
                d[v] = d[v] + 1

    return F, S


def connect_disjoint_subtrees(G, F, S):
    """
    Add edges to maximally leafy forest F to make it a spanning tree T of G
    :param F:
    :return: T
    """
    edges_difference = G.edges() - F.edges()

    # remove edges that connect vertices from same disjoint subtree
    edges_to_add = edges_difference.copy()
    for e in edges_difference:
        u, v = e[0], e[1]
        if S[u] == S[v]:
            edges_to_add.discard(e)

    # sort edges in ascending order by weight
    edges_to_add = sorted(list(edges_to_add), key=lambda x: G.edges[x]['weight'])

    # add edges using Kruskal's Algorithm
    T = F.copy()
    for e in edges_to_add:
        u, v = e[0], e[1]
        if S[u] != S[v]:
            T.add_edge(*e)
            S.union(u, v)
            if T.number_of_edges() == G.number_of_nodes() - 1:
                break

    return T

def prune_leaves(T):
    """
    Greedily prunes the leaves of tree T ONLY if it reduces the average pairwise distance
    :param T:
    :return:
    """
    # TODO: need to check that pruning a leaf will decrease pairwise distance
    T_pruned = T.copy()
    for v in T.nodes:
        if T.degree[v] == 1 and T_pruned.number_of_nodes() > 1:
            T_pruned.remove_node(v)
    return T_pruned


def k_star(G):
    """

    :param G:
    :return:
    """
    pass


def visualize_results(G, F, T, T_pruned):
    """
    Visualizes input graph and output tree side by side for easy comparison

    :param G: input Graph
    :param T: output Tree
    :return:
    """

    plt.subplot(141)
    nx.draw(G, with_labels=True)
    plt.subplot(142)
    nx.draw(F, with_labels=True)
    plt.subplot(143)
    nx.draw(T, with_labels=True)
    plt.subplot(144)
    nx.draw(T_pruned, with_labels=True)

    plt.subplot(141).set_title('Graph')
    plt.subplot(142).set_title('Forest')
    plt.subplot(143).set_title('Tree')
    plt.subplot(144).set_title('Pruned Tree')
    plt.show()


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]

    just_testing_single_graph = False

    if just_testing_single_graph:
        # path = "phase1_input_graphs\\25.in"
        # path = "inputs\small-249.in"
        path = "inputs\large-15.in"
        G = read_input_file(path)
        T = solve(G, visualize=True, verbose=True)
        assert is_valid_network(G, T)
        print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        write_output_file(T, 'out/test.out')
    else:
        # output_dir = "experiment_outputs/test1"
        output_dir = "phase2_outputs"
        input_dir = "inputs"
        pairwise_distances = np.array([])
        for input_path in os.listdir(input_dir):
            graph_name = input_path.split(".")[0]
            G = read_input_file(f"{input_dir}/{input_path}")

            # Solve problem and time the elapsed time
            start_time = time.time()
            T = solve(G)
            elapsed_time = time.time() - start_time

            assert is_valid_network(G, T)

            cost = average_pairwise_distance(T)
            pairwise_distances = np.append(pairwise_distances, cost)
            # print("Finished solving:", graph_name)
            # print('With total runtime:', elapsed_time, "(s)")
            print("With average  pairwise distance: {}".format(cost), "\n")

            # write_output_file(T, f"{output_dir}/{graph_name}.out")

        print("Average of all scores:", np.mean(pairwise_distances))
        # TODO: write a save_to_csv function that saves a table of inputs and their runtime
