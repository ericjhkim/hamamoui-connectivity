"""
This project is a reproduction of ideas presented in Hamaoui's paper:
Connectivity Maintenance through Unlabeled Spanning Tree Matching
https://doi.org/10.1007/s10846-024-02048-9

Created on Sun Dec 01 2022

@author: ericjhkim
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import tools
import h5py
from networkx.algorithms.tree.mst import SpanningTreeIterator, random_spanning_tree, number_of_spanning_trees
from networkx.algorithms import isomorphism
from datetime import datetime

# Simulation constants
N = 8
N_TRIALS = 50
N_SIMULATION = 100
P_THRESH = 0.1

# Settings
SAVE = True
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():

    # Results for running each SET of phi values N_SIMULATION times
    matching_frequencies = []
    phis = np.linspace(0, 1, 11)

    for s in range(N_SIMULATION):
        print("-"*50)
        print(f"Simulation {s+1} of {N_SIMULATION}")
        # Results for running each phi value N_TRIALS times
        results = []
        for phi in phis:
            G1 = nx.erdos_renyi_graph(N, phi)
            G2 = nx.erdos_renyi_graph(N, phi)

            count = 0
            for _ in range(N_TRIALS):
                try:
                    T1 = random_spanning_tree(G1)
                    belongs = check_tree_dynamically(T1, G2)
                    if belongs:
                        count += 1
                except:
                    continue

            results.append(count/N_TRIALS)

        matching_frequencies.append(results)
        print(f"Simulation {s+1}: {results}")

    # Calculate probability that matching frequency is over threshold
    final = np.zeros(len(phis))
    for col in range(len(phis)):
        for row in matching_frequencies:
            if row[col] >= P_THRESH:
                final[col] += 1
    final /= N_SIMULATION

    if SAVE:
        with h5py.File(f'data/data_{TIMESTAMP}_{N}_{int(P_THRESH*100)}.h5', 'w') as f:
            f.create_dataset('N', data=N)
            f.create_dataset('P_THRESH', data=P_THRESH)
            f.create_dataset('phis', data=phis)
            f.create_dataset('matching_frequencies', data=matching_frequencies)
            f.create_dataset('pr', data=final)

    # Plot the results
    tools.plot_data(N, P_THRESH, phis, final)

def is_tree_isomorphic(tree1, tree2):
    """Check if two trees are isomorphic."""
    gm = isomorphism.GraphMatcher(tree1, tree2)
    return gm.is_isomorphic()

def check_tree_dynamically(T1, G2):
    """
    Check if a spanning tree T1 (of G1) is isomorphic to any spanning tree of another graph (G2)
    without precomputing all spanning trees of G2.
    """
    edges = list(G2.edges())
    for edge_subset in itertools.combinations(edges, len(G2.nodes) - 1):
        candidate_tree = G2.edge_subgraph(edge_subset).copy()
        if nx.is_tree(candidate_tree):
            if is_tree_isomorphic(T1, candidate_tree):
                return True
    return False

def draw_graph(nx_graph):
    fig, axes = plt.subplots(1,1,dpi=72)
    nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), ax=axes, with_labels=True)
    plt.show()

if __name__ == "__main__":
    main()