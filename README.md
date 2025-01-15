# Hamamoui: Graph Connectivity Maintenance Implementation

This is an implementation of theory presented in Hamamoui's paper [1], maintaining graph connectivity through the matching of unlabled spanning trees.

## Results

![Matching Frequencies](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/fm_by_order.png)

_The probability of an isomorphic match increases with graph order._

## Simulation

![Simulation](https://github.com/ericjhkim/hamamoui-connectivity/blob/main/visualizations/anim_20250114_213707.gif)

_Using a Laplacian-based linear PD controller, the agents converge to the target locations (red markers) while maintaing connectivity (green connecting lines) of the matching spanning tree._

## References
  1. Hamaoui, M. (2024). *Connectivity Maintenance through Unlabeled Spanning Tree Matching*. J Intell Robot Syst 110, 15 [doi:10.1007/s10846-024-02048-9](https://doi.org/10.1007/s10846-024-02048-9)