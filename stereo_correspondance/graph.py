import numpy as np
import maxflow
import energy
import cv2


class AlphaExpansion:

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
        self.best_cut = np.inf

    def save_disparity_map(self):
        f = self.f.reshape((self.h, self.w))
        cv2.imwrite("output/disparity.png", f)

    def initialize_labeling_function(self):
        f = np.random.randint(low=0, high=self.labels.shape[0], size=(self.h, self.w))
        for i in range(self.labels.shape[0]):
            f[f == i] = self.labels[i]

        return f

    def get_labeling_from_partition(self, G, label):
        f = np.copy(self.f)
        nodes = np.arange(0, f.shape[0])
        reachable = G.get_grid_segments(nodes)
        f[reachable] = label
        return f

    def construct_graph(self, label):

        # todo: initialize with better params
        G = maxflow.Graph[int](2, 2)
        G.add_nodes(self.f.shape[0])

        # data term will apply a penalty if a pixel in L does not correspond to a pixel in R, for the disparity label
        print("Adding data")
        G = self.add_data_edges(G, label)
        # smoothness term is used is used to penalize pixels that are close to one another, but have a different label
        print("Adding smoothness")
        # G = self.add_smoothness_edges(G, label)

        return G

    def D_p(self, p, label):
        # find the best match within the label range, clipped at twenty
        THRESHOLD = 20
        p_index = np.unravel_index(p, (self.h, self.w))

        I_p = self.L[p_index]

        left = p_index[1] + np.max([label - (self.increment // 2), 0])
        right = p_index[1] + np.min([label + (self.increment // 2), self.w - 1])

        if left > self.w - 1:
            return THRESHOLD

        pixel_values = self.R[p_index[0], left:right]

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
                from_source_cap = self.D_p(pixel, label)
                if pixel_label == label:
                    G.add_tedge(pixel, from_source_cap, 1000000)
                else:
                    G.add_tedge(pixel, from_source_cap, self.D_p(pixel, pixel_label))

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
            nodes = G.add_nodes(1)
            G.add_edge(p, nodes[0], self.V(f_p, alpha, p, q, interpolate=True), 0)
            G.add_edge(nodes[0], q, self.V(alpha, f_q, p, q, interpolate=True), 0)
            # G.add_edge(a, self.not_alpha, capacity=self.V(f_p, f_q, p, q))
            G.add_tedge(nodes[0], 0, self.V(f_p, f_q, p, q))

        return G

    def add_smoothness_edges(self, G, alpha):
        # assumes 4 neighboring edges (left, top, right, bottom)
        pixel = 0
        # a = self.f.shape[0]
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
        flow = G.maxflow()
        return flow, G

    def calculate_best_alpha_expansion(self):
        self.save_disparity_map()
        # current_energy = energy.calculate_energy(self.f, self.L, self.R)
        print("current energy", self.best_cut)

        labels = self.labels
        # np.array([, 55, 75, 100]) #
        cut_value_array = np.zeros_like(labels)
        partition_list = []

        for i in range(labels.shape[0]):
            label = labels[i]
            cut_value, partition = self.alpha_expansion(label)
            cut_value_array[i] = cut_value
            partition_list.append(partition)

        min_cut_value_index = np.argmin(cut_value_array)
        best_cut_value = cut_value_array[min_cut_value_index]
        print("best cut index", min_cut_value_index)
        best_partition = partition_list[min_cut_value_index]

        f_prime = self.get_labeling_from_partition(best_partition, labels[min_cut_value_index])

        # energy_after_expansion = energy.calculate_energy(f_prime, self.L, self.R)
        print("energy after expansion", best_cut_value)

        has_lowered_energy = False
        if best_cut_value < self.best_cut:
            self.best_cut = best_cut_value
            has_lowered_energy = True

        return has_lowered_energy, f_prime

    def calculate_disparity_map(self):

        has_expansion_reduced_energy = True

        while has_expansion_reduced_energy:
            has_expansion_reduced_energy, f = self.calculate_best_alpha_expansion()
            if has_expansion_reduced_energy:
                self.f = f

        return self.f


