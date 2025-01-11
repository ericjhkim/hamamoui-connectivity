# Implementation of Olfati-Saber's paper: Flocking for Multi-Agent Dynamic Systems
#
# Created by Eric Kim (07-11-2024)

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tools_flocking as tools
from datetime import datetime
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.algorithms import isomorphism
import itertools
import matplotlib.pyplot as plt

# Constants
N_AGENTS = 6                                # Number of agents
SENSOR_RANGE = 20                           # Distance at which agents can sense each other
SIM_TIME = 10                               # Simulation time in seconds
dt = 0.1                                    # Simulation interval
DIMS = 3                                    # Number of dimensions

# Directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

# Controls
SAVE_DATA = False
CREATE_GIF = True
SEED = 42
np.random.seed(SEED)

# Initial conditions
def main():
    print("Starting simulation...")
    frame_count = 0

    swarm = Agents(DIMS,N_AGENTS,SENSOR_RANGE,dt)
    t = 0

    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

    with tqdm(total=SIM_TIME) as pbar:
        while t < SIM_TIME:
            A = swarm.get_adjacency(swarm.states)
            for i in range(N_AGENTS):
                u = swarm.get_command(i)
                swarm.update(u,i)

            t = round(t + dt, 10)
            pbar.update(min(dt, SIM_TIME - pbar.n))

            # Plot the swarm
            if CREATE_GIF:
                last = t+dt > SIM_TIME
                tools.create_frame(swarm, A, t, frame_count, last, swarm.data[0], swarm.tgt_Q)
                frame_count += 1

    # Calculate adjacency adherence
    fraction_adjacency = swarm.get_adjacency_obedience()
    # Adjacency obedience (ie. did the swarm stay connected as per the matching tree? Lower number = connectivity was better maintained)
    print(swarm.A_T1-fraction_adjacency)

    # Save data
    if SAVE_DATA:
        tools.save_to_h5py(np.array(swarm.data), filename=f"sim_data_{TIMESTAMP}", dataset_name="simulation")

    # Create a GIF using PIL
    if CREATE_GIF:
        frames = []
        frame_files = [f'frames/frame_{i:04d}.png' for i in range(frame_count)]
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=dt*1000, loop=0)

        # Clean up the frame images after the GIF is created
        import shutil
        shutil.rmtree('frames')

    tools.plot_data(np.array(swarm.data),swarm.data[0],swarm.tgt_Q)
    tools.plot_separation(np.array(swarm.data))

class Agents:
    def __init__(self, DIMS, N_AGENTS, SENSOR_RANGE, dt):
        self.DIMS = DIMS
        self.N_AGENTS = N_AGENTS
        self.r = SENSOR_RANGE
        self.dt = dt

        # Initial graph
        self.states = np.concatenate((self.generate_3d_coordinates(N_AGENTS, 5, SENSOR_RANGE),np.zeros((self.DIMS,N_AGENTS))),axis=0)
        self.A1 = self.get_adjacency(self.states)
        self.G1 = nx.from_numpy_array(self.A1)
        tools.plot_agents_3d(self.states,self.A1)

        # Target graph
        self.tgt_Q = np.concatenate((self.generate_3d_coordinates(N_AGENTS, 5, SENSOR_RANGE),np.zeros((self.DIMS,N_AGENTS))),axis=0)
        self.A2 = self.get_adjacency(self.tgt_Q)
        self.G2 = nx.from_numpy_array(self.A2)
        tools.plot_agents_3d(self.tgt_Q,self.A2)

        # Check match
        found = False
        for _ in range(50):
            try:
                T1 = random_spanning_tree(self.G1,seed=SEED)
                match, T2 = self.check_tree_dynamically(T1, self.G2)
                if match:
                    found = True
                    break
            except:
                continue
        
        if found:
            print("Found a matching tree")
            self.draw_graphs([self.G1, T1, self.G2, T2])
            matcher = nx.isomorphism.GraphMatcher(T1, T2)
            matches = matcher.is_isomorphic() # Run this before the mapping
            self.mapping = matcher.mapping
            print("Node Mapping from T1 to T2:", self.mapping, matches)

            # Get adjacency matrix of T1
            self.A_T1 = nx.adjacency_matrix(T1).toarray()

            # Calculate T2 Laplacian
            self.L = self.compute_laplacian(T1.edges())
        else:
            print("No matching tree found")

        self.data = [np.array(self.states)]

    def compute_laplacian(self, edges):
        """
        Compute the graph Laplacian for the spanning tree.
        """
        L = np.zeros((self.N_AGENTS, self.N_AGENTS))
        for i, j in edges:
            L[i, i] += 1
            L[j, j] += 1
            L[i, j] -= 1
            L[j, i] -= 1
        return L

    def get_command(self, i):
        """
        Compute the control command for agent i with Laplacian, target-seeking, and PD control.
        """
        # Current position and velocity of agent i
        q_i = self.states[:self.DIMS, i]         # Position
        v_i = self.states[self.DIMS:, i]    # Velocity
        
        # Target position for agent i
        k = self.mapping[i]
        q_k = self.tgt_Q[:self.DIMS, k]

        # Laplacian term (connectivity maintenance)
        neighbors_influence = -self.L[i, :] @ self.states[:self.DIMS, :].T  # Shape: (DIMS,)

        # Target-seeking term
        target_term = q_k - q_i

        # Control command (combined terms)
        k1 = 0.2  # Laplacian term weight
        k2 = 1.0  # Target-seeking weight
        k_d = 0.7  # Damping term weight
        u = k1 * neighbors_influence + k2 * target_term - k_d * v_i

        return u

    def draw_graph(self, nx_graph):
        fig, axes = plt.subplots(1,1,dpi=72)
        nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), ax=axes, with_labels=True)
        plt.show()

    def draw_graphs(self, graphs):
        fig, axes = plt.subplots(1,len(graphs),dpi=72)
        for g, graph in enumerate(graphs):
            nx.draw(graph, pos=nx.spring_layout(graph), ax=axes[g], with_labels=True)
        plt.show()

    def check_tree_dynamically(self, T1, G2):
        """
        Check if a spanning tree T1 (of G1) is isomorphic to any spanning tree of another graph (G2)
        without precomputing all spanning trees of G2.
        """
        edges = list(G2.edges())
        for edge_subset in itertools.combinations(edges, len(G2.nodes) - 1):
            candidate_tree = G2.edge_subgraph(edge_subset).copy()
            if nx.is_tree(candidate_tree):
                if self.is_tree_isomorphic(T1, candidate_tree):
                    return True, candidate_tree
        return False, None
    
    def is_tree_isomorphic(self, tree1, tree2):
        """Check if two trees are isomorphic."""
        gm = isomorphism.GraphMatcher(tree1, tree2)
        return gm.is_isomorphic()

    def generate_3d_coordinates(self, N, d_min, SENSOR_RANGE):
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

        return np.transpose(coordinates)

    def get_adjacency(self, states):
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(states[:self.DIMS,i] - states[:self.DIMS,j]) <= self.r:
                    A[i,j] = 1
                if i == j:
                    A[i,j] = 0
        return A

    def update(self,u,i_agent):
        self.states[self.DIMS:, i_agent] += u*self.dt           # Update velocity
        self.states[:self.DIMS, i_agent] += self.states[self.DIMS:, i_agent]*self.dt
        self.save_data()

    def save_data(self):
        self.data.append(np.array(self.states))

    def get_adjacency_obedience(self):
        A_hist = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for t in range(len(self.data)):
            A = self.get_adjacency(self.data[t])
            for i in range(N_AGENTS):
                for j in range(N_AGENTS):
                    if i != j:
                        if self.A_T1[i,j] == 1:
                            if A[i,j] == 1:
                                A_hist[i,j] += 1

        return A_hist/len(self.data)

if __name__ == '__main__':
    main()