import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import os

"""
CS170 Sp2020 Project: Cell Tower Network Design for Horizon Wireless


authors: Benjamin Chang, Kelvin Pang 
date: 4/29/2020
"""

def solve(G, visualize=False, verbose=False):
    """
    Solves the problem statement by computing the minimum over the following possible solutions:
        - G (smart pruned)
        - Maximum Spanning Tree (smart pruned)
        - Minimum Spanning Tree (smart pruned)
        -
    :param G: networkx.Graph
    :return: T networkx.Graph
    """
    # TODO: your code here!

    F, S = maximally_leafy_forest(G)
    T = connect_disjoint_subtrees(G, F, S)
    T_pruned = prune_leaves(T, smart_pruning=True)

    # visualize and compare results
    if visualize:
        visualize_results(G, F, T, T_pruned, include_edge_weights=True)
    if verbose:
        MST = nx.maximum_spanning_tree(G)
        print("Cost of G:", average_pairwise_distance(G))
        print("Cost of an MST:", average_pairwise_distance(MST))
        print("Cost of T:", average_pairwise_distance(T))
        print("Cost of T_pruned:", average_pairwise_distance(T_pruned))

    #
    # all_solutions = []
    # all_solutions.append(T_pruned)
    # all_solutions.append(T_pruned)

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

    # Branch from vertices with highest degree first
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)

    for v, deg in sorted_nodes:
        S_prime = {}
        d_prime = 0
        neighbors = list(G.neighbors(v))
        sorted_neighbors_asc = sorted(neighbors, key=lambda x: G[x][v]['weight'])
        sorted_neighbors_desc = sorted(neighbors, key=lambda x: G[x][v]['weight'], reverse=True)
        for u in neighbors:
            if S[u] != S[v] and S[u] not in S_prime.values():
                d_prime = d_prime + 1
                S_prime[u] = S[u]
        if d[v] + d_prime >= 3:
            for u in S_prime.keys():
                F.add_edge(u, v, weight=G[u][v]['weight'])
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
    edges_to_add = sorted(list(edges_to_add), key=lambda x: G[x[0]][x[1]]['weight'])

    # add edges using Kruskal's Algorithm
    T = F.copy()
    for e in edges_to_add:
        u, v = e[0], e[1]
        if S[u] != S[v]:
            T.add_edge(u, v, weight=G[u][v]['weight'])
            S.union(u, v)
            if T.number_of_edges() == G.number_of_nodes() - 1:
                break

    return T

def prune_leaves(T, smart_pruning=True):
    """
    Greedily prunes the leaves of tree T ONLY if it reduces the average pairwise distance
    :param T:
    :return:
    """

    T_pruned = T.copy()

    for v in T.nodes:
        if T.degree[v] == 1 and T_pruned.number_of_nodes() > 1:
            if smart_pruning is False:
                T_pruned.remove_node(v)
            else:
                T_pruned_check = T_pruned.copy()
                T_pruned_check.remove_node(v)
                if average_pairwise_distance_fast(T_pruned_check) < average_pairwise_distance_fast(T_pruned):
                    T_pruned.remove_node(v)

    return T_pruned


def k_star(G):
    """

    :param G:
    :return:
    """
    pass


def visualize_results(G, F, T, T_pruned, include_edge_weights=False):
    """
    Visualizes input graph and output tree side by side for easy comparison

    :param G: input Graph
    :param T: output Tree
    :return:
    """
    if include_edge_weights is True:
        plt.subplot(141)
        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

        plt.subplot(142)
        pos = nx.spring_layout(F)
        nx.draw(F, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(F, pos=pos, edge_labels=nx.get_edge_attributes(F, 'weight'))

        plt.subplot(143)
        pos = nx.spring_layout(T)
        nx.draw(T, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(T, pos=pos, edge_labels=nx.get_edge_attributes(T, 'weight'))

        plt.subplot(144)
        pos = nx.spring_layout(T_pruned)
        nx.draw(T_pruned, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(T_pruned, pos=pos, edge_labels=nx.get_edge_attributes(T_pruned, 'weight'))
    else:
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

    # Seed Random Datapoint Selection
    seed_val = 420
    np.random.seed(seed_val)

    just_testing_single_graph = False

    if just_testing_single_graph:
        # path = "phase1_input_graphs\\25.in"
        # path = "inputs\small-249.in"
        path = "inputs\small-8.in"
        G = read_input_file(path)
        start_time = time.time()
        T = solve(G, visualize=True, verbose=True)
        elapsed_time = time.time() - start_time
        assert is_valid_network(G, T)
        print('Total runtime:', elapsed_time, "(s)")
        print("Average  pairwise distance: {}".format(average_pairwise_distance_fast(T)))
        # write_output_file(T, 'out/test.out')
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

            cost = average_pairwise_distance_fast(T)
            pairwise_distances = np.append(pairwise_distances, cost)
            print("Finished solving:", graph_name)
            print('Total runtime:', elapsed_time, "(s)")
            print("Average pairwise distance: {}".format(cost), "\n")

            # write_output_file(T, f"{output_dir}/{graph_name}.out")

        print("Average Cost of all scores:", np.mean(pairwise_distances))
        # TODO: write a save_to_csv function that saves a table of inputs and their runtime
