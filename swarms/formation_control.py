"""
This project is a reproduction of ideas presented in Hamaoui's paper:
Connectivity Maintenance through Unlabeled Spanning Tree Matching
https://doi.org/10.1007/s10846-024-02048-9

Created on Wed Dec 04 2022

@author: ericjhkim
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.algorithms import isomorphism
import itertools

# Example Usage
N = 6                   # Number of agents
d_min = 5               # Minimum distance between agents
SENSOR_RANGE = 20       # Maximum distance
phi = 0.7
seed = 42
np.random.seed(seed)
N_TRIALS = 50

def main():

    ## Initial coordinates
    c1 = generate_3d_coordinates(N, d_min, 30)
    # print(c1)

    A1 = compute_adjacency_matrix(c1, SENSOR_RANGE)
    # print(A1)

    G1 = nx.from_numpy_array(A1)
    # draw_graph(G1)

    # plot_agents_3d(c1, A1)

    ## Target coordinates
    np.random.seed(seed+45)
    c2 = generate_3d_coordinates(N, d_min, 30)
    # print(c2)

    A2 = compute_adjacency_matrix(c2, SENSOR_RANGE)
    # print(A1)

    G2 = nx.from_numpy_array(A2)
    # draw_graph(G2)

    # plot_agents_3d(c2, A2)

    found = False
    for _ in range(N_TRIALS):
        try:
            T1 = random_spanning_tree(G1,seed=seed)
            match, T2 = check_tree_dynamically(T1, G2)
            if match:
                found = True
                break
        except:
            continue
    
    if found:
        print("Found a matching tree")
        draw_graphs([G1, T1, G2, T2])
        print(T1.edges())
        matcher = nx.isomorphism.GraphMatcher(T1, T2)
        matches = matcher.is_isomorphic() # Run this before the mapping
        mapping = matcher.mapping
        print("Node Mapping from T1 to T2:", mapping, matches)
    else:
        print("No matching tree found")

    plot_multi_agents_3d(c1, A1, c2, A2, mapping)
    
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
                return True, candidate_tree
    return False, None

def graph_to_3d_positions(graph, d_min, SENSOR_RANGE):
    N = len(graph.nodes)
    coordinates = np.random.uniform(0, SENSOR_RANGE, size=(N, 3))

    for _ in range(200):  # Iterative relaxation to adjust positions
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue

            node_pos = coordinates[node]
            neighbor_positions = coordinates[neighbors]

            # Compute centroid of neighbors
            centroid = neighbor_positions.mean(axis=0)

            # Adjust node position toward the centroid
            displacement = 0.1 * (centroid - node_pos)
            new_position = node_pos + displacement

            # Enforce distance constraints
            distances = np.linalg.norm(coordinates - new_position, axis=1)
            if np.all((distances >= d_min) & (distances <= SENSOR_RANGE)):
                coordinates[node] = new_position

        # Ensure nodes stay within SENSOR_RANGE
        coordinates = np.clip(coordinates, 0, SENSOR_RANGE)

    return coordinates

def draw_graph(nx_graph):
    fig, axes = plt.subplots(1,1,dpi=72)
    nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), ax=axes, with_labels=True)
    plt.show()

def draw_graphs(graphs):
    fig, axes = plt.subplots(1,len(graphs),dpi=72)
    for g, graph in enumerate(graphs):
        nx.draw(graph, pos=nx.spring_layout(graph), ax=axes[g], with_labels=True)
    plt.show()

def generate_3d_coordinates(N, d_min, SENSOR_RANGE):
    def is_valid_point(new_point, points):
        if len(points) == 0:
            return True
        distances = np.linalg.norm(points - new_point, axis=1)
        return np.all((distances >= d_min) & (distances <= SENSOR_RANGE))

    coordinates = []

    # Create points iteratively
    while len(coordinates) < N:
        new_point = np.random.uniform(0, SENSOR_RANGE, size=3)
        if is_valid_point(new_point, np.array(coordinates)):
            coordinates.append(new_point)

    return np.array(coordinates)


def compute_adjacency_matrix(coordinates, SENSOR_RANGE):
    N = len(coordinates)
    adjacency_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= SENSOR_RANGE:
                    adjacency_matrix[i, j] = 1

    return adjacency_matrix

def plot_agents_3d(coordinates, adjacency_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    # Plot the points with labels
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='b', marker='o')
        ax.text(xi, yi, zi, f'{i}', color='red')

    # Draw connections based on adjacency matrix
    N = len(coordinates)
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency_matrix[i, j] == 1:
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='gray', linestyle='--')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Agent Positions with Connections')

    plt.show()

def plot_multi_agents_3d(coords1, adj1, coords2, adj2, mapping):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colours = ["b","r"]

    for j,coordinates in enumerate([coords1,coords2]):
        adjacency_matrix = [adj1,adj2][j]

        # Extract x, y, z coordinates
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]

        # Plot the points with labels
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.scatter(xi, yi, zi, c=colours[j], marker='o')
            ax.text(xi, yi, zi, f'{i}', color='red')

        # Draw connections based on adjacency matrix
        N = len(coordinates)
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency_matrix[i, j] == 1:
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='gray', linestyle='--')

    # Plot command vectors:
    for i in range(len(coords1)):
        print(coords1[i],coords2[i])
        k = mapping[i]
        ax.plot([coords1[i][0], coords2[k][0]], [coords1[i][1], coords2[k][1]], [coords1[i][2], coords2[k][2]], c='green', linestyle='-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Agent Positions with Connections')

    plt.show()

if __name__ == "__main__":
    main()