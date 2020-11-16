import maxflow
import numpy as np

print("maxflow")

g = maxflow.Graph[int](2, 2)
# Add two (non-terminal) nodes. Get the index to the first one.
nodes = g.add_nodes(2)
nodes_2 = g.add_nodes(1)
# Create two edges (forwards and backwards) with the given capacities.
# The indices of the nodes are always consecutive.
g.add_edge(nodes[0], nodes[1], 1, 2)
# g.add_edge(nodes[0], nodes[1], 1, 2)
# g._add_edges(
#     charar,
#     np.array([nodes[1]], dtype=np.uint32),
#     np.array([1], dtype=np.int),
#     np.array([2], dtype=np.int))
# Set the capacities of the terminal edges...
# ...for the first node.
g.add_tedge(nodes[0], 2, 5)
# ...for the second node.
g.add_tedge(nodes[1], 9, 4)

flow = g.maxflow()
print("Maximum flow:", flow)

print("Segment of the node 0:", g.get_segment(nodes[0]))
print("Segment of the node 1:", g.get_segment(nodes[1]))

s = g.get_grid_segments(np.array([0, 1]))
print()