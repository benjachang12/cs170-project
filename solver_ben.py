import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import random
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

    F, S = maximally_leafy_forest(G)
    leafyT = connect_disjoint_subtrees(G, F, S)
    leafyT_pruned = prune_leaves(leafyT, smart_pruning=True)

    F_asc, S_asc = maximally_leafy_forest(G, neighbor_sort="ascending")
    leafyT_asc = connect_disjoint_subtrees(G, F_asc, S_asc)
    leafyT_asc_pruned = prune_leaves(leafyT_asc, smart_pruning=True)

    F_desc, S_desc = maximally_leafy_forest(G, neighbor_sort="descending")
    leafyT_desc = connect_disjoint_subtrees(G, F_desc, S_desc)
    leafyT_desc_pruned = prune_leaves(leafyT_desc, smart_pruning=True)

    minST = nx.minimum_spanning_tree(G)
    maxST = nx.maximum_spanning_tree(G)
    minST_pruned = prune_leaves(minST, smart_pruning=True)
    maxST_pruned = prune_leaves(maxST, smart_pruning=True)


    # Take the minimum over all these different approaches
    all_solutions = [leafyT_pruned, leafyT_asc_pruned, leafyT_desc_pruned, minST_pruned, maxST_pruned]
    all_costs = []
    for tree in all_solutions:
        all_costs.append(average_pairwise_distance_fast(tree))
    min_solution = all_solutions[all_costs.index(min(all_costs))]

    # visualize and compare results
    if visualize:
        visualize_results(G, F, leafyT, leafyT_pruned, include_edge_weights=True)
    if verbose:
        print("Cost of G:", average_pairwise_distance_fast(G))
        print("Cost of leafyT:", average_pairwise_distance_fast(leafyT))
        print("Cost of leafyT_pruned:", average_pairwise_distance_fast(leafyT_pruned))
        print("Cost of MinST:", average_pairwise_distance_fast(minST))
        print("Cost of MaxST:", average_pairwise_distance_fast(maxST))
        print("Cost of MinST_pruned:", average_pairwise_distance_fast(minST_pruned))
        print("Cost of MaxST_pruned:", average_pairwise_distance_fast(maxST_pruned))

    return min_solution


def brute_force_search(G):
    """

    :param G:
    :return:
    """


def maximally_leafy_forest(G, neighbor_sort=None):
    """
    Computes the maximally leafy forest F for G
    A maximally leafy forest F is a set of disjointly "leafy" subtrees of G,
    where F is not a subgraph of any other leafy forest of G
    :param G:
    :param neighbor_sort:
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

        if neighbor_sort is not None:
            if neighbor_sort is "ascending":
                neighbors = sorted(neighbors, key=lambda x: G[x][v]['weight'])
            elif neighbor_sort is "descending":
                neighbors = sorted(neighbors, key=lambda x: G[x][v]['weight'], reverse=True)
            elif neighbor_sort is "random":
                neighbors = random.shuffle(neighbors)

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
    :param G:
    :param F:
    :param S:
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
    :param smart_pruning:
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
    random.seed(seed_val)

    generate_outputs = False
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

            if generate_outputs:
                write_output_file(T, f"{output_dir}/{graph_name}.out")

        print("Average Cost of all scores:", np.mean(pairwise_distances))
        # TODO: write a save_to_csv function that saves a table of inputs and their runtime
