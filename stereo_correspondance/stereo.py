

def pairwise_stereo(left, right, method):
    if method == "ssd":
        print("calculating stereo map ssd")
        return pairwise_stereo_ssd(left, right)
    elif method == "graph_cut":
        print("calculating stereo map graph cut")
        return pairwise_stereo_graph_cut(left, right)


def pairwise_stereo_ssd(left, right):

    return None


def pairwise_stereo_graph_cut(left, right):

    return None

