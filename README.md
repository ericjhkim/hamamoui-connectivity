# Hamamoui: Graph Connectivity Maintenance Implementation

This is an implementation of theory presented in Hamamoui's paper [1], maintaining graph connectivity through the matching of unlabled spanning trees.

## Graph Results
### Instructions
To run the probabilistic matching tree simulation, run [graphs/tree_matching.py](graphs/tree_matching.py).
Corresponding data will be stored in [data/data_graphs](data/data_graphs).

### Results
![Matching Frequencies](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/visualizations/fm_by_order.png)
_The probability of an isomorphic match increases with graph order._

## Swarming Results
### Instructions
To run the swarm simulation, run [swarms/swarms.py](swarms/swarms.py).
Corresponding data will be stored in [data/data_swarms](data/data_swarms).

### Results
Consider a system of agents with initial formation graph G1 and target formation graph G2, a matching spanning tree T1 (in terms of G1 node labels), and the corresponding tree T2 (in terms of G2 node labels).
![Graphs](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/visualizations/matching_graphs_example_1.png)
_Note that T1 and T2 are isomorphic, differing only in node labels._

The swarm's behaviour as it moves towards the target formation positions can be observed below.

![Simulation Overshoot](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/visualizations/anim_20250212_074531.gif)
_Using a Laplacian-based linear PD controller, the agents converge to the target locations (red markers) while maintaing connectivity (green connecting lines) of the matching spanning tree. Note that in this case, the gain of the Laplacian term is insufficient, as broken connections appear (red lines)._

With a different tuning, we can observe different behaviour.

![Simulation Undershoot](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/visualizations/anim_20250212_074853.gif)
_Note that in this case, the gain of the Laplacian term is sufficient, preventing connections from breaking, but agents do not reach their target positions._

## References
  1. Hamaoui, M. (2024). *Connectivity Maintenance through Unlabeled Spanning Tree Matching*. J Intell Robot Syst 110, 15 [doi:10.1007/s10846-024-02048-9](https://doi.org/10.1007/s10846-024-02048-9)