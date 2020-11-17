import numpy as np
import maxflow
import cv2


class AlphaExpansion:

    def __init__(self, left, right, labels, lambda_v):
        self.L = left.astype(np.float)
        self.R = right.astype(np.float)
        self.labels = labels
        self.h = left.shape[0]
        self.w = left.shape[1]
        self.n = self.h * self.w
        self.d_low = labels[0]
        self.d_high = labels[labels.shape[0]-1]
        self.assignment_table = self.initialize_assignment_function()
        self.lambda_v = lambda_v

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

    def initialize_assignment_function(self):

        a_table = np.zeros((self.n+self.d_high, self.n))

        f = np.random.randint(low=self.d_low, high=self.d_high, size=self.n)
        p_indices = np.indices((1, self.n))[1, :, :]
        q_indices = np.copy(p_indices) + f

        a_table[q_indices, p_indices] = 1
        return a_table

    def get_f_from_assignment_table(self):
        f_indices = np.where(self.assignment_table)
        f = np.zeros((self.h, self.w))
        for i, j in zip(f_indices[0], f_indices[1]):
            index = np.unravel_index(j, (self.h, self.w))
            disparity = i - j
            f[index] = disparity
            print(i, j)
        return f

    def get_labeling_from_partition(self, G, label, A_0, A_alpha):
        A_prime = []
        # todo
        # f = np.copy(self.f)
        # nodes = np.arange(0, f.shape[0])
        # reachable = G.get_grid_segments(nodes)
        # f[np.where(reachable)] = label
        return A_prime

    def calculate_expansion_tables(self, alpha):
        A_0 = np.copy(self.assignment_table)
        A_alpha = np.copy(self.assignment_table)

        return A_0, A_alpha

    def construct_graph(self, alpha, A_0, A_alpha):

        # todo: initialize with better params
        G = maxflow.Graph[int](2, 2)

        assignment_edges = np.zeros_like(self.assignment_table)
        assignment_vertex_labels = np.zeros_like(self.assignment_table)

        assignment_indices = np.where(self.assignment_table)
        number_of_vertices = assignment_indices[0].shape[0]
        if number_of_vertices > self.n:
            raise RuntimeError("Too many assignments")
        G.add_nodes(assignment_indices[0].shape[0])
        a = 0
        for q, p in zip(assignment_indices[0], assignment_indices[1]):
            assignment_vertex_labels[q, p] = a
            a += 1

        for q, p in zip(assignment_indices[0], assignment_indices[1]):
            a = assignment_vertex_labels[q, p]
            s_a_cost = self.get_s_a_cost(p, q, alpha, A_0, A_alpha, assignment_vertex_labels)
            a_t_cost = self.get_a_t_cost(p, q, alpha, A_0, A_alpha, assignment_vertex_labels)
            G.add_tedge(a, s_a_cost, a_t_cost)
            # an assignment a'=<p',q'> is a neighbor to another assignment a=<p,q> if
            # p is as neighbor of p' or q is a neighbor of q'
            G = self.add_neighborhood_edges(p, G, assignment_edges, A_0, A_alpha, assignment_vertex_labels)
            G = self.add_neighborhood_edges(q, G, assignment_edges, A_0, A_alpha, assignment_vertex_labels)
            G = self.add_expansion_edges(p, G, A_0, A_alpha, assignment_vertex_labels)

        return G

    def add_occlusion_edges(self, G, alpha):

        assignment_indices = np.where(self.assignment_table)
        for q, p in zip(assignment_indices[0], assignment_indices[1]):
            disparity = q - p
            N_p = np.sum(self.assignment_table[:, p])
            cost = 0
            if N_p < 1:
                cost = self.lambda_v
            if disparity == alpha:
                G.add_tedge(p, cost, 0)
            else:
                G.add_tedge(p, 0, cost)
        return G

    def D_p(self, p, q):
        # find the best match within the label range, clipped at thresh
        THRESHOLD = self.d_thresh
        p_index = np.unravel_index(p, (self.h, self.w))
        q_index = np.unravel_index(q, (self.h, self.w))

        if q[1] > self.w - 1:
            return THRESHOLD

        L_I_p = self.L[p_index, :]
        R_I_p = self.R[q_index, :]
        diff = L_I_p - R_I_p
        squared = diff ** 2
        value = np.sqrt(np.sum(squared))
        if value > THRESHOLD:
            return THRESHOLD
        return value

    def D_occ_p(self, p, A):
        matches = np.sum(A[:,p])
        if matches == 1:
            return self.lambda_v
        return 0.0

    def D_occ_a(self, p, q, A):
        return self.D_occ_p(p, A) + self.D_occ_p(q, A)

    def get_s_a_cost(self, p, q, alpha, A_0, A_alpha, assignment_vertex_labels):
        a_0 = A_0[q, p]
        if a_0 == 1:
            return self.D_occ_a(p, q, A_0+A_alpha)

        return self.D_a(p, q)

    def get_a_t_cost(self, p, q, alpha, A_0, A_alpha, assignment_vertex_labels):
        a_alpha = A_alpha[q, p]
        if a_alpha == 1:
            return self.D_occ_a(p, q, A_0+A_alpha)

        return self.D_a(p, q) + self.D_smooth(p, q)

    def D_smooth(self, p, q):
        # todo
        return 0.0

    def D_a(self, p, q):
        # todo
        return 0.0

    def V(self):
        # todo
        return 0.0

    def add_neighborhood_edges(self, p, G, assignment_edges, A_0, A_alpha, assignment_vertex_labels):
        # todo
        # iterate through neighbors of p
        # if q < self.n + self.d_high - 1 and p < self.n - 1:
        #     # <q_2,p_2> has same disparity of <q,p>
        #     q_2 = q + 1
        #     p_2 = p + 1
            # is_assigned = self.assignment_table[q_2, p_2]
            # if assignment is different, add penalty

        return G

    def add_expansion_edges(self, p, G, A_0, A_alpha, assignment_vertex_labels):
        matches = np.where(A_0[:, p])[0].shape
        if matches == 0:
            return G

        q_0 = np.where(A_0[:, p])[0][0]
        matches = np.where(A_alpha[:, p])[0].shape
        if matches == 0:
            return G

        q_alpha = np.where(A_alpha[:, p])[0][0]

        if q_alpha < p or q_0 < p:
            raise RuntimeError("error in assignment")

        a1 = assignment_vertex_labels[q_0, p]
        a2 = assignment_vertex_labels[q_alpha, p]

        G.add_edge(a1, a2, 10000000, self.lambda_v)
        return G

    def alpha_expansion(self, alpha, A_0, A_alpha):

        G = self.construct_graph(alpha, A_0, A_alpha)
        flow = G.maxflow()
        return flow, G

    def calculate_alpha_expansion(self):
        self.save_disparity_map()
        current_energy = self.calculate_energy(self.assignment_table)
        print("current energy", current_energy)

        labels = self.labels

        arr = np.arange(labels.shape[0])
        np.random.shuffle(arr)
        for i in arr:
            label = labels[i]
            A_0, A_alpha = self.calculate_expansion_tables(label)
            cut_value, partition = self.alpha_expansion(label)
            A_prime = self.get_labeling_from_partition(partition, label, A_0, A_alpha)
            energy_after_expansion = self.calculate_energy(A_prime)
            print("energy after expansion", energy_after_expansion, label)
            if energy_after_expansion < current_energy:
                return True, A_prime

        return False, None

    def calculate_disparity_map(self):

        has_expansion_reduced_energy = True

        while has_expansion_reduced_energy:
            has_expansion_reduced_energy, A_prime = self.calculate_alpha_expansion()
            if has_expansion_reduced_energy:
                self.assignment_table = A_prime

        f = self.get_f_from_assignment_table()
        return f

    """
    ------------------------------ energy functions
    """
    def calculate_energy(self, a_table):
        smooth = self.calculate_smoothness_energy(a_table)
        data = self.calculate_data_energy(a_table)
        occ = self.calculate_occlusion_energy(a_table)

        return data + smooth + occ

    def calculate_occlusion_energy(self, a_table):
        non_occluded_pixels = np.sum(a_table)
        occluded_pixels = self.n - non_occluded_pixels

        if occluded_pixels < 0:
            raise RuntimeError("occluded pixel data error")

        return non_occluded_pixels * self.lambda_v

    def calculate_data_energy(self, a_table):
        pixel = 0
        sum = 0.0
        # todo

        return sum

    def calculate_smoothness_energy(self, f):
        pixel = 0
        sum = 0.0
        # todo

        return sum


