import numpy as np
import maxflow
import cv2


class AlphaExpansion:

    def __init__(self, left, right, labels, k=10, k_not=5, increment=5, v_thresh=65, d_thresh=20):
        self.L = left.astype(np.float)
        self.R = right.astype(np.float)
        self.labels = labels
        self.h = left.shape[0]
        self.w = left.shape[1]
        self.f = self.initialize_labeling_function().flatten()
        # k and k_not are potts model constants
        self.K = k
        self.K_not = k_not
        # threshold to determine when to use k or k_not for smoothness term
        self.intensity_thresh = v_thresh
        self.d_thresh = d_thresh
        # range to compare disparity with
        self.increment = increment

    # used to track how disparity changes after each expansion iteration
    def save_disparity_map(self):
        f = np.copy(self.f.reshape((self.h, self.w)))
        f = f * 9
        # l = 255.0 / (len(self.labels) + 1)
        # s = l
        # for label in self.labels:
        #     f[f==label] = s
        #     s += l

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
        f[np.where(reachable)] = label
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
        G = self.add_smoothness_edges(G, label)

        return G

    def D_p(self, p, label):
        # find the best match within the label range, clipped at twenty
        THRESHOLD = self.d_thresh
        p_index = np.unravel_index(p, (self.h, self.w))

        increment = self.increment
        I_p = self.L[p_index[0], p_index[1], :]

        left = p_index[1] + np.max([label - increment, 0])
        right = p_index[1] + np.min([label + increment, self.w - 1])

        if left > self.w - 1:
            return THRESHOLD

        I_left = ((self.R[p_index[0], left, :] + I_p) / 2) - I_p
        I_right = I_left
        if left > self.w - 1:
            I_right = ((self.R[p_index[0], right, :] + I_p) / 2) - I_p

        pixel_values = self.R[p_index[0], left:right, :]

        diff = pixel_values - I_p
        ssd = np.sqrt(np.sum(diff ** 2, axis=1))
        ssd_left = np.sqrt(np.sum(I_left ** 2))
        ssd_right = np.sqrt(np.sum(I_right ** 2))
        value = THRESHOLD
        if ssd.shape[0] > 0:
            value = np.min(ssd)
        value = np.min(np.array([value, ssd_left, ssd_right]))
        if value > THRESHOLD:
            return THRESHOLD

        return value

    def add_data_edges(self, G, label):

        pixel = 0
        for i in range(self.h):
            # if i % 50 == 0:
            #     print(i)
            for j in range(self.w):
                pixel_label = self.f[pixel]
                from_source_cap = self.D_p(pixel, label)
                if pixel_label == label:
                    G.add_tedge(pixel, from_source_cap, 100000000)
                else:
                    G.add_tedge(pixel, from_source_cap, self.D_p(pixel, pixel_label))

                pixel += 1

        return G

    def V(self, p_label, q_label, p, q):
        if p_label == q_label:
            return 0.0

        p_index = np.unravel_index(p, (self.h, self.w))
        q_index = np.unravel_index(q, (self.h, self.w))

        diff = self.L[p_index[0], p_index[1], :] - self.L[q_index[0], q_index[1], :]

        intensity_diff = np.sqrt(np.sum(diff ** 2))

        if intensity_diff > self.intensity_thresh:
            return self.K_not

        return self.K

    def add_neighborhood_edges(self, G, p, q, alpha):

        f_p = self.f[p]
        f_q = self.f[q]

        if f_p == f_q:
            return G
        else:
            # create new intermediate node a and three new edges
            p_q_dist = self.V(f_p, alpha, p, q)
            a_q_dist = self.V(alpha, f_q, q, p)
            nodes = G.add_nodes(1)
            G.add_edge(p, nodes[0], p_q_dist, p_q_dist)
            G.add_edge(q, nodes[0], a_q_dist, a_q_dist)
            G.add_tedge(nodes[0], 0, self.V(f_p, f_q, p, q))

        return G

    def add_smoothness_edges(self, G, alpha):
        # assumes 4 neighboring edges (left, top, right, bottom)
        pixel = 0
        # a = self.f.shape[0]
        for i in range(self.h):
            for j in range(self.w):
                if i < self.h - 1:
                    bottom_pixel = pixel + self.w
                    G = self.add_neighborhood_edges(G, pixel, bottom_pixel, alpha)

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
        current_energy = self.calculate_energy(self.f)
        print("current energy", current_energy)

        labels = self.labels
        # np.array([, 55, 75, 100]) #
        cut_value_array = np.zeros_like(labels)
        partition_list = []
        best_f_prime = None
        has_lowered_energy = False

        arr = np.arange(labels.shape[0])
        np.random.shuffle(arr)
        # range(labels.shape[0])
        for i in arr:
            label = labels[i]
            cut_value, partition = self.alpha_expansion(label)
            cut_value_array[i] = cut_value
            partition_list.append(partition)
            f_prime = self.get_labeling_from_partition(partition, label)
            energy_after_expansion = self.calculate_energy(f_prime)
            print("energy after expansion", energy_after_expansion, label)
            if energy_after_expansion < current_energy:
                has_lowered_energy = True
                current_energy = energy_after_expansion
                best_f_prime = f_prime
                break

        return has_lowered_energy, best_f_prime

    def calculate_disparity_map(self):

        has_expansion_reduced_energy = True

        while has_expansion_reduced_energy:
            has_expansion_reduced_energy, f = self.calculate_best_alpha_expansion()
            if has_expansion_reduced_energy:
                self.f = f

        return self.f

    """
    ------------------------------ energy functions
    """
    def calculate_energy(self, f):
        s = self.calculate_smoothness_energy(f)
        d = self.calculate_data_energy(f)

        return d + s

    def calculate_data_energy(self, f):
        pixel = 0
        sum = 0.0
        for i in range(self.h):
            for j in range(self.w):
                sum += self.D_p(pixel, f[pixel])
                pixel += 1
        return sum

    def calculate_smoothness_energy(self, f):
        pixel = 0
        sum = 0.0
        for i in range(self.h):
            for j in range(self.w):

                if i < self.h - 1:
                    bottom_pixel = pixel + self.w
                    sum += self.V(f[pixel], f[bottom_pixel], pixel, bottom_pixel)

                if j < self.w - 1:
                    right_pixel = pixel + 1
                    sum += self.V(f[pixel], f[right_pixel], pixel, right_pixel)
                pixel += 1
        return sum


