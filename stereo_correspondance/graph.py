








# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.minimum_cut.html
def get_edges_from_st_cut(reachable, non_reachable, G):
    cut_set = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cut_set.update((u, v) for v in nbrs if v in non_reachable)


# constructs a graph with nodes and edges and edge weights as discussed in the paper
# according to the alpha beta swap method
def construct_alpha_beta_graph_from_labeling_function(f, left, right, labels):

    G = nx.DiGraph()

    return 0.0


def expand_graph(G, label_pair):

    H = G.copy()

    return H


def calculate_new_labeling(G, edge_cut_set):
    f_prime = 0

    return f_prime


# performs a multi-way cut as described in the paper
def multiway_cut(G, labels):
    # set of edges that define the cut
    edge_cut_set = []
    cut_value_array = np.zeros_like(labels)

    for label in labels:

        G_prime = expand_graph(G, label)

        cut_value, partition = nx.minimum_cut(G, "s", "t")

    return edge_cut_set


# Return true if an alpha-beta swap lowered the energy
# Otherwise return false
# def alpha_beta_swap_iteration(f, left, right, labels):
#     energy_value = energy.calculate_energy(f, left, right)
#     G = construct_alpha_beta_graph_from_labeling_function(f, left, right, labels)
#
#     edge_cut_set = multiway_cut(G, labels)
#
#     f_prime = calculate_new_labeling(G, edge_cut_set)
#
#     new_energy_value = energy.calculate_energy(f, left, right)
#
#     found_decreasing_cut = False
#     if new_energy_value < energy_value:
#         found_decreasing_cut = True
#
#     return found_decreasing_cut, f_prime
#
#
# def alpha_beta_swap(left, right, labels):
#     h = left.shape[0]
#     w = left.shape[1]
#     # f: pixel --> label
#     f = energy.initialize_labeling_function((h, w), labels)
#
#     is_energy_decreasing = True
#
#     while is_energy_decreasing:
#         is_energy_decreasing, f_prime = alpha_beta_swap_iteration(f, left, right, labels)
#         if is_energy_decreasing:
#             f = f_prime
#
#     return f

