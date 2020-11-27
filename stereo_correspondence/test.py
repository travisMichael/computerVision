import networkx as nx
import numpy as np
import cv2

d = cv2.imread("input_images/cones/disp2.png", 0)
d = cv2.imread("output/disparity_ssd_map.png", 0)

G = nx.DiGraph()

nodes = np.array([0, 1])

# [0, 1, 2, 3]

G.add_nodes_from(nodes)
G.add_nodes_from(["s", "t"])

# edges = np.array([
#     [0, 1, 2, 3],
#     [1, 2, 3, 1]
# ])

# [(1, 2), (1, 3)]
# add_weighted_edges_from
G.add_edge("s", 0, capacity=2.0)
G.add_edge("s", 1, capacity=9.0)

G.add_edge(0, 1, capacity=1.0)
G.add_edge(1, 0, capacity=2.0)

G.add_edge(0, "t", capacity=5.0)
G.add_edge(1, "t", capacity=4.0)

# edges = minimum_st_edge_cut(G, "s", "t")


cut_value, partition = nx.minimum_cut(G, "s", "t")
reachable, non_reachable = partition

cut_set = set()
for u, nbrs in ((n, G[n]) for n in reachable):
    cut_set.update((u, v) for v in nbrs if v in non_reachable)

G.remove_edges_from(cut_set)

print("hello")