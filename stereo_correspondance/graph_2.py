import numpy as np
import networkx as nx
import energy


def get_labeling_from_parition(best_partition):
    return 0.0


class AlphaExpansion:
    # two terminal nodes for performing a min-st cut
    alpha = "s"
    not_alpha = "t"

    def __init__(self, left, right, labels, k=20):
        self.L = left.astype(np.float)
        self.R = right.astype(np.float)
        self.labels = labels
        self.h = left.shape[0]
        self.w = left.shape[1]
        self.f = self.initialize_labeling_function().flatten()
        # K = potts model constant
        self.K = k
        self.increment = labels[1] - labels[0]

    def initialize_labeling_function(self):
        f = np.random.randint(low=0, high=self.labels.shape[0], size=(self.h, self.w))
        for i in range(self.labels.shape[0]):
            f[f == i] = self.labels[i]

        return f

    def construct_graph(self, label):

        G = nx.DiGraph()

        # data term will apply a penalty if a pixel in L does not correspond to a pixel in R, for the disparity label
        print("Adding data")
        G = self.add_data_edges(G, label)
        # smoothness term is used is used to penalize pixels that are close to one another, but have a different label
        print("Adding smoothness")
        G = self.add_smoothness_edges(G, label)


        return G

    def D_p(self, pixel, label):
        # find the best match within the label range, clipped at twenty
        THRESHOLD = 20
        i, j = pixel

        I_p = self.L[i, j]

        left = j + np.max([label - (self.increment // 2), 0])
        right = j + np.min([label + (self.increment // 2), self.w - 1])

        if left > self.w - 1:
            return THRESHOLD

        pixel_values = self.L[i, left:right]

        abs_diff = abs(pixel_values - I_p)
        value = np.min(abs_diff)
        if value > THRESHOLD:
            return THRESHOLD

        return value

    def add_data_edges(self, G, label):

        pixel = 0
        for i in range(self.h):
            if i % 50 == 0:
                print(i)
            for j in range(self.w):
                pixel_label = self.f[pixel]
                if pixel_label == label:
                    G.add_edge(pixel, self.not_alpha, capacity=np.inf)
                else:
                    G.add_edge(pixel, self.not_alpha, capacity=self.D_p((i, j), pixel_label))

                G.add_edge(self.alpha, pixel, capacity=self.D_p((i, j), label))
                pixel += 1

        return G

    def V(self, p_label, q_label, p, q, interpolate=False):
        if p_label == q_label:
            return 0.0

        p_index = np.unravel_index(p, (self.h, self.w))
        q_index = np.unravel_index(q, (self.h, self.w))

        term_2 = self.L[q_index]
        if interpolate:
            term_2 = float(self.L[p_index] - self.L[q_index]) / 2.0

        intensity_diff = abs(self.L[p_index] - term_2)

        if intensity_diff > 5:
            return self.K

        return self.K * 2

    def add_neighborhood_edges(self, G, p, q, alpha):

        f_p = self.f[p]
        f_q = self.f[q]

        if f_p == f_q:
            pass
            # G.add_edge(p, q, capacity=self.V(p, q))
        else:
            # create new intermediate node a and three new edges
            a = str(p) + "_" + str(q)
            G.add_edge(p, a, capacity=self.V(f_p, alpha, p, q, interpolate=True))
            G.add_edge(a, q, capacity=self.V(alpha, f_q, p, q, interpolate=True))
            G.add_edge(a, self.not_alpha, capacity=self.V(f_p, f_q, p, q))

        return G

    def add_smoothness_edges(self, G, alpha):
        # assumes 4 neighboring edges (left, top, right, bottom)
        pixel = 0
        for i in range(self.h):
            if i % 50 == 0:
                print(i)
            for j in range(self.w):
                if i > 0:
                    top_pixel = pixel - self.w
                    G = self.add_neighborhood_edges(G, pixel, top_pixel, alpha)

                if i < self.h - 1:
                    bottom_pixel = pixel + self.w
                    G = self.add_neighborhood_edges(G, pixel, bottom_pixel, alpha)

                if j > 0:
                    left_pixel = pixel - 1
                    G = self.add_neighborhood_edges(G, pixel, left_pixel, alpha)

                if j < self.w - 1:
                    right_pixel = pixel + 1
                    G = self.add_neighborhood_edges(G, pixel, right_pixel, alpha)

                pixel += 1

        return G

    def alpha_expansion(self, alpha):

        G = self.construct_graph(alpha)

        # 1715567
        cut_value, partition = nx.minimum_cut(G, self.alpha, self.not_alpha)

        return cut_value, partition

    def calculate_best_alpha_expansion(self):
        current_energy = energy.calculate_energy(self.f, self.L, self.R)

        cut_value_array = np.zeros_like(self.labels)
        partition_list = []

        for i in range(self.labels.shape[0]):
            label = self.labels[i]
            cut_value, partition = self.alpha_expansion(label)
            cut_value_array[i] = cut_value
            partition_list.append(partition)

        min_cut_value_index = np.min(cut_value_array)
        best_partition = partition_list[min_cut_value_index]

        f_prime = get_labeling_from_parition(best_partition)

        energy_after_expansion = energy.calculate_energy(self.f, self.L, self.R)

        has_lowered_energy = False
        if energy_after_expansion < current_energy:
            has_lowered_energy = True

        return has_lowered_energy, f_prime

    def calculate_disparity_map(self):

        has_expansion_reduced_energy = True

        while has_expansion_reduced_energy:
            has_expansion_reduced_energy, f = self.calculate_best_alpha_expansion()
            if has_expansion_reduced_energy:
                self.f = f

        return self.f


