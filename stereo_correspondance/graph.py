import networkx as nx


# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.minimum_cut.html
def get_edges_from_st_cut(reachable, non_reachable, G):
    cut_set = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cut_set.update((u, v) for v in nbrs if v in non_reachable)


def alpha_beta_swap(f, labels):

    pass