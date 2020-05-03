import networkx as nx
from networkx.algorithms import approximation
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import os

# Parallelization Imports
from joblib import Parallel, delayed
import multiprocessing

"""
CS170 Sp2020 Project: Cell Tower Network Design for Horizon Wireless

authors: Benjamin Chang, Kelvin Pang 
date: 4/29/2020
"""

# Counter Global Variables
COUNT_LEAFYT = 0
COUNT_MINST = 0
COUNT_MAXST = 0
COUNT_VC = 0
COUNT_DS = 0


def solve(G, alpha_range=np.arange(0,1001,10), verbose=False, parallel=False, count=False):
    """
    Solves the problem statement by computing the minimum over the following possible solutions:
        - G (smart pruned)
        - Maximum Spanning Tree (smart pruned)
        - Minimum Spanning Tree (smart pruned)
        -
    :param G: networkx.Graph
    :return: T networkx.Graph
    """

    # Generate leafyT solutions (with root priority heuristics and varying values of alpha)
    # Note: when alpha = 0, this is equivalent to no root priority heuristic
    def maximally_leafy_tree(alpha):
        F, S = maximally_leafy_forest(G, neighbor_sort="ascending", use_root_priority=True, alpha=alpha)
        leafyT = connect_disjoint_subtrees(G, F, S)
        leafyT_pruned = prune_leaves(leafyT, smart_pruning=True)
        return leafyT_pruned

    all_solutions = []
    all_costs = []

    if parallel:
        num_cores = multiprocessing.cpu_count()
        all_solutions = Parallel(n_jobs=num_cores)(delayed(maximally_leafy_tree)(alpha) for alpha in alpha_range)
        all_costs = Parallel(n_jobs=num_cores)(delayed(average_pairwise_distance_fast)(sol) for sol in all_solutions)
    else:
        for alpha in alpha_range:
            sol = maximally_leafy_tree(alpha)
            all_costs.append(average_pairwise_distance_fast(sol))
            all_solutions.append(sol)

    leafyT_pruned = all_solutions[0]

    if verbose:
        # Plot heuristic hyperparameter tuning graph
        plt.figure()
        plt.plot(alpha_range, all_costs)
        plt.xlabel("alpha"), plt.ylabel("Cost")
        plt.title("Cost vs Alpha (neighbors sorted ascending)")
        plt.show(block=False)

    # Generate pruned minST and maxST solutions
    minST = nx.minimum_spanning_tree(G)
    maxST = nx.maximum_spanning_tree(G)
    minST_pruned = prune_leaves(minST, smart_pruning=True)
    maxST_pruned = prune_leaves(maxST, smart_pruning=True)
    all_solutions.append(minST_pruned)
    all_solutions.append(maxST_pruned)
    all_costs.append(average_pairwise_distance_fast(minST_pruned))
    all_costs.append(average_pairwise_distance_fast(maxST_pruned))

    # Generate min vertex cover spanning tree solution
    minVCST = min_vertex_cover_spanning_tree(G)
    all_solutions.append(minVCST)
    all_costs.append(average_pairwise_distance_fast(minVCST))

    # Generate min dominating set spanning tree solution
    minDSST = min_dominating_set_spanning_tree(G)
    all_solutions.append(minDSST)
    all_costs.append(average_pairwise_distance_fast(minDSST))

    # Take the minimum over all these different approaches
    min_solution = all_solutions[all_costs.index(min(all_costs))]

    # Visualize and compare results (only first leafyT shown, when alpha=0)
    if verbose:
        # Print costs of solutions
        leafyT_pruned = all_solutions[0]
        print("Cost of G:", average_pairwise_distance_fast(G))
        print("Cost of leafyT_pruned:", average_pairwise_distance_fast(leafyT_pruned))
        print("Cost of MinST_pruned:", average_pairwise_distance_fast(minST_pruned))
        print("Cost of MaxST_pruned:", average_pairwise_distance_fast(maxST_pruned))
        print("Cost of MinVCST:", average_pairwise_distance_fast(minVCST))
        print("Cost of MinDSST:", average_pairwise_distance_fast(minDSST))

        # Visualize graphs of solutions
        visualize_graph(G, title="Input Graph")
        visualize_graph(leafyT_pruned, title="Pruned Leafy Tree")
        visualize_graph(minST_pruned, title="Pruned MinST")
        visualize_graph(maxST_pruned, title="Pruned MaxST")
        visualize_graph(minVCST, title="MinVCST")
        visualize_graph(minDSST, title="MinDSST")

    # Keep a counter of which solutions were chosen as minimum
    if count:
        global COUNT_LEAFYT, COUNT_MINST, COUNT_MAXST, COUNT_VC, COUNT_DS
        if min_solution == minST_pruned:
            COUNT_MINST = COUNT_MINST + 1
        elif min_solution == maxST_pruned:
            COUNT_MAXST = COUNT_MAXST + 1
        elif min_solution == minVCST:
            COUNT_VC = COUNT_VC + 1
        elif min_solution == minDSST:
            COUNT_DS = COUNT_DS + 1
        else:
            COUNT_LEAFYT = COUNT_LEAFYT + 1

    return min_solution


def maximally_leafy_forest(G, neighbor_sort=None, use_root_priority=False, alpha=1):
    """
    Computes the maximally leafy forest F for G
    A maximally leafy forest F is a set of disjointly "leafy" subtrees of G,
    where F is not a subgraph of any other leafy forest of G
    :param G: input graph
    :param neighbor_sort: order neighbors are added to subtree (None, "ascending", "descending", or "random")
    :param use_root_priority:
    :param alpha: hyperparameter for root priority (high alpha = higher priority for low average edge cost)
    :return: F networkx.Graph
             S disjoint subtrees
    """
    # create graph with same nodes as G, but zero edges
    F = nx.Graph()
    F.add_nodes_from(G.nodes)

    S = nx.utils.UnionFind()
    d = np.zeros(len(G.nodes))
    for v in G.nodes:
        S[v]

    def root_priority_heuristic(x, G, alpha):
        """
        Outputs the priority of choosing a vertex as the root of a subtree in the maximally leafy forest.
        Build the maximally leafy forest using vertices with highest priority first
        :param x: x is tuple (vertex, deg)
        :param G: input graph
        :param alpha: tunable hyperparameter
        :return:
        """
        v = x[0]
        deg = x[1]
        average_edge_cost = 0
        for u in G.neighbors(v):
            average_edge_cost += G[u][v]['weight']
        if deg != 0:
            average_edge_cost = average_edge_cost / deg
        priority = deg + alpha/average_edge_cost
        return priority

    if not use_root_priority:
        # Branch from vertices with highest degree first
        sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    else:
        # Branch from vertices according to root priority heuristic
        sorted_nodes = sorted(G.degree, key=lambda x: root_priority_heuristic(x, G, alpha), reverse=True)

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
                random.shuffle(neighbors)

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


def min_vertex_cover_spanning_tree(G):
    """
    Generate a min vertex cover approximation, then form minimum spanning tree of these nodes
    Minimum vertex cover is from networkx's implementation, with an approximation ratio of 2
    :param G:
    :return:
    """
    minVC = nx.Graph()
    if G.number_of_nodes() == 1:
        minVC.add_nodes_from(G.nodes)
    else:
        minVC.add_nodes_from(approximation.min_weighted_vertex_cover(G))
    S = nx.utils.UnionFind()
    for v in minVC.nodes:
        S[v]
    minVCST = connect_disjoint_subtrees(G, minVC, S)
    return minVCST


def min_dominating_set_spanning_tree(G):
    """
    Generate a min dominating set approximation, then form minimum spanning tree of these nodes
    Minimum dominating set is from networkx's implementation of minimum cardinality edge dominating set,
    with an approximation ratio of 2
    :param G:
    :return:
    """
    minDS = nx.Graph()
    if G.number_of_nodes() == 1:
        minDS.add_nodes_from(G.nodes)
    S = nx.utils.UnionFind()
    edges_to_add = approximation.min_edge_dominating_set(G)
    for e in edges_to_add:
        u, v = e[0], e[1]
        S.union(S[v], S[u])
        minDS.add_edge(u, v, weight=G[u][v]['weight'])
    minDSST = connect_disjoint_subtrees(G, minDS, S)
    return minDSST


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


def visualize_graph(G, title="untitled", layout="spring_layout"):
    """
    Visualizes the graph using networkx's drawing tools (uses matplotlib)
    :param G: input graph to draw
    :param title: graph title
    :param layout: layout type
    :return:
    """
    plt.figure()
    plt.title(title)
    if layout is "spring_layout":
        pos = nx.spring_layout(G)
    elif layout is "planar_layout":
        pos = nx.planar_layout(G)
    elif layout is "random_layout":
        pos = nx.random_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.show(block=False)


# Here's an example of how to run your solver.
# Usage: python3 solver.py test.in
if __name__ == '__main__':

    ###########################################
    #     Solver Settings (CHANGE ME)         #
    ###########################################
    test_single_graph = False           # True if testing only single input, False if testing ALL inputs
    generate_outputs = False            # True if generating outputs for submission
    track_solution_count = True         # True if keep track of solution type counter
    alpha_range = np.arange(0, 3001, 5)      # Range of alpha values to iterate over for hyperparameter tuning
    ###########################################

    # Seed Random Datapoint Selection
    seed_val = 420
    random.seed(seed_val)

    if test_single_graph:
        # path = "phase1_input_graphs\\25.in"
        path = "inputs\\small-25.in"
        G = read_input_file(path)

        # Parallel Solver
        start_time = time.time()
        T_parallel = solve(G, alpha_range=alpha_range, parallel=True)
        elapsed_time = time.time() - start_time
        print("Total runtime (Parallel):", elapsed_time, "(s)")

        # Regular Solver
        start_time = time.time()
        T = solve(G, alpha_range=alpha_range)
        elapsed_time = time.time() - start_time
        print('Total runtime (Regular):', elapsed_time, "(s)")

        # Rerun Regular Solver with verbose outputs
        T = solve(G, alpha_range=alpha_range, verbose=True, count=track_solution_count)

        assert is_valid_network(G, T_parallel)
        assert is_valid_network(G, T)

        print("Average pairwise distance (Parallel): {}".format(average_pairwise_distance_fast(T_parallel)))
        print("Average pairwise distance (Regular): {}".format(average_pairwise_distance_fast(T)))

    else:
        # output_dir = "experiment_outputs/test1"
        output_dir = "phase2_outputs"
        input_dir = "inputs"

        start_time = time.time()
        pairwise_distances = np.array([])

        for input_path in os.listdir(input_dir):
            graph_name = input_path.split(".")[0]
            G = read_input_file(f"{input_dir}/{input_path}")
            T = solve(G, alpha_range=alpha_range, parallel=True, count=track_solution_count)
            assert is_valid_network(G, T)

            cost = average_pairwise_distance_fast(T)
            pairwise_distances = np.append(pairwise_distances, cost)
            print("Finished solving:", graph_name)
            print("Average pairwise distance: {}".format(cost), "\n")

            if generate_outputs:
                write_output_file(T, f"{output_dir}/{graph_name}.out")

        elapsed_time = time.time() - start_time
        print('Total runtime:', elapsed_time, "(s)")
        print("Average Cost of all scores:", np.mean(pairwise_distances))

    print("SOLUTION COUNTER:")
    print("LeafyT:", COUNT_LEAFYT, "MinST:", COUNT_MINST, "MaxST:", COUNT_MAXST, "VC:", COUNT_VC, "DS:", COUNT_DS)
    plt.show()