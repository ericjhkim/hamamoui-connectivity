"""
This project is a reproduction of ideas presented in Hamaoui's paper:
Connectivity Maintenance through Unlabeled Spanning Tree Matching
https://doi.org/10.1007/s10846-024-02048-9

This code applies the probabilistic matching tree algorithm to a swarm of agents.

@author: ericjhkim
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tools_swarms as tools
from datetime import datetime
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.algorithms import isomorphism
import itertools

# Constants
N_AGENTS = 8                                # Number of agents

SENSOR_RANGE = 25                           # Distance at which agents can sense each other
D_MIN = 5                                   # Minimum distance between agents (for random generation of initial position)
D_MAX = SENSOR_RANGE*1.5                    # Maximum distance between agents (for random generation of initial position)
T_VEC = [20.0, 20.0, 0.0]                   # Translation vector (for random generation of initial position)

SIM_TIME = 10                               # Simulation time in seconds
dt = 0.1                                    # Simulation interval
DIMS = 3                                    # Number of dimensions

# Directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

# Controls
SAVE_DATA = True
CREATE_GIF = True
SEED = 1
np.random.seed(SEED)

# Initial conditions
def main():
    print("Starting simulation...")

    # Initialize swarm object
    swarm = Agents(N_AGENTS,SENSOR_RANGE,T_VEC,dt)

    # Run simulation
    t = 0.0
    frame_count = 0
    with tqdm(total=SIM_TIME, desc="Running Simulation", unit="steps") as pbar:
        while t < SIM_TIME:

            A = swarm.get_adjacency(swarm.states)
            for i in range(N_AGENTS):
                u = swarm.get_command(i)
                swarm.update(u,i)

            swarm.save_data()

            t = round(t + dt, 10)
            pbar.update(min(dt, SIM_TIME - pbar.n))

    # Generate gif frames
    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

        frame_count = 0
        states = swarm.data
        lims = tools.get_lims(states)
        for i in tqdm(range(len(states)), desc="Generating GIF Frames", unit="frame"):
            A = swarm.get_adjacency(states[i])

            tools.create_frame(states[i], A, swarm.A_T1, i*dt, frame_count, False, states[0], swarm.tgt_Q, lims)
            frame_count += 1

    # Calculate adjacency adherence
    fraction_adjacency = swarm.get_adjacency_obedience()
    print("Adjacency obedience: \n",swarm.A_T1-fraction_adjacency)

    # Save data
    if SAVE_DATA:
        tools.save_to_h5py(np.array(swarm.data), filename=f"data/data_swarms/data_{TIMESTAMP}", dataset_name="simulation")

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

    # Data plotting
    tools.plot_data(np.array(swarm.data),swarm.data[0],swarm.tgt_Q)
    tools.plot_separation(np.array(swarm.data))

class Agents:
    def __init__(self, N_AGENTS, SENSOR_RANGE, T_VEC, dt):
        
        self.N_AGENTS = N_AGENTS
        self.SENSOR_RANGE = SENSOR_RANGE                    # Sensor range
        self.dt = dt
        self.d_min = D_MIN                                  # Minimum distance between agents (for random generation)
        self.d_max = D_MAX                                  # Maximum distance between agents (for random generation)
        self.t_vec = T_VEC

        # Initial graph
        self.states = np.concatenate((self.generate_3d_coordinates(N_AGENTS, self.d_min, self.d_max),np.zeros((3,N_AGENTS))),axis=0)
        self.A1 = self.get_adjacency(self.states)
        self.G1 = nx.from_numpy_array(self.A1)
        tools.plot_agents_3d(self.states,self.A1)

        # Target graph
        self.tgt_Q = np.concatenate((self.generate_3d_coordinates(N_AGENTS, self.d_min, self.d_max, translation=self.t_vec),np.zeros((3,N_AGENTS))),axis=0)
        self.A2 = self.get_adjacency(self.tgt_Q)
        self.G2 = nx.from_numpy_array(self.A2)
        tools.plot_agents_3d(self.tgt_Q,self.A2)

        # Check match
        found = False
        for _ in range(50):
            try:
                T1 = random_spanning_tree(self.G1, seed=SEED)
                match, T2 = self.check_tree_dynamically(T1, self.G2)
                if match:
                    found = True
                    break
            except:
                continue
        
        if found:
            print("Found a matching tree.")
            matcher = nx.isomorphism.GraphMatcher(T1, T2)
            matches = matcher.is_isomorphic() # Run this before the mapping
            self.mapping = matcher.mapping
            print("Node Mapping from T1 to T2: ", self.mapping, " | Is isomorphic: ", matches)

            self.A_T1 = nx.adjacency_matrix(T1, nodelist=sorted(T1.nodes()), dtype=np.float64).toarray()
            # print(f"G1's Adjacency Matrix:\n{self.A1}")
            # print(f"G1 Subgraph's Adjacency Matrix:\n{self.A_T1}")

            self.A_T2 = nx.adjacency_matrix(T2, nodelist=sorted(T2.nodes()), dtype=np.float64).toarray()
            # print(f"G2's Adjacency Matrix:\n{self.A2}")
            # print(f"G2 Subgraph's Adjacency Matrix:\n{self.A_T2}")

            tools.draw_graphs([self.G1, T1, self.G2, T2])
            # tools.plot_agents_connectivity(self.states, self.A1, self.A_T1)
        else:
            print("No matching tree found")

        self.data = [np.array(self.states)]

    def compute_laplacian(self, A):
        """
        Compute the Laplacian matrix from a given adjacency matrix.
        """
        D = np.diag(A.sum(axis=1))  # Compute the degree matrix
        L = D - A  # Laplacian matrix
        return L

    def get_command(self, i):
        """
        Compute the control command for agent i with Laplacian, target-seeking, and PD control.
        """
        # Current position and velocity of agent i
        q_i = self.states[:3, i]        # Position
        v_i = self.states[3:, i]        # Velocity
        
        # Target position for agent i
        k = self.mapping[i]
        q_k = self.tgt_Q[:3, k]

        # Laplacian term (connectivity maintenance)
        L = self.compute_laplacian(self.get_adjacency(self.states))
        neighbors_influence = -L[i,:] @ self.states[:3, :].T    # Shape: (DIMS,)

        # Target-seeking term
        target_term = q_k - q_i

        # Control command (combined terms)
        k1 = 0.01                           # Laplacian term weight
        k2 = 1.0                            # Target-seeking weight
        k_d = 0.8                           # Damping term weight
        u = k1 * neighbors_influence + k2 * target_term - k_d * v_i

        return u

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

    def generate_3d_coordinates(self, N, d_min, d_max, translation=[0,0,0]):
        """
        Generate N 3D coordinates with a minimum distance of d_min and a maximum distance of d_max.
        This is to nondeterministically initialize agents' locations.
        """
        def is_valid_point(new_point, points):
            if len(points) == 0:
                return True
            distances = np.linalg.norm(points - new_point, axis=1)
            return np.all((distances >= d_min) & (distances <= d_max))

        coordinates = []

        # Create points iteratively
        while len(coordinates) < N:
            new_point = np.random.uniform(0, d_max, size=3)
            if is_valid_point(new_point, np.array(coordinates)):
                new_point += translation
                coordinates.append(new_point)

        return np.transpose(coordinates)

    def get_adjacency(self, states):
        """
        Compute the adjacency matrix for a given set of agent states.
        """
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(states[:3,i] - states[:3,j]) <= self.SENSOR_RANGE:
                    A[i,j] = 1
                if i == j:
                    A[i,j] = 0
        return A

    def update(self,u,i_agent):
        """
        Update the state of agent i based on the control command u.
        """
        self.states[3:, i_agent] += u*self.dt           # Update velocity
        self.states[:3, i_agent] += self.states[3:, i_agent]*self.dt

    def save_data(self):
        """
        Store state in data storage object
        """
        self.data.append(np.array(self.states))

    def get_adjacency_obedience(self):
        """
        Used to calculate adjacency obedience (ie. did the swarm stay connected as per the matching tree? Lower number = connectivity was better maintained)
        """
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