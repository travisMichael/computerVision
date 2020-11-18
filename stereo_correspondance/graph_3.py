import numpy as np
import maxflow
import cv2


class AlphaExpansion:

    def __init__(self, left, right, labels, lambda_v, d_thresh):
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
        self.d_thresh = d_thresh

    # returns the assigned pixel to p
    def get_q(self, p, alpha):
        p_index = np.unravel_index(p, (self.h, self.w))
        if p_index[1] + alpha > self.w - 1:
            return p

        q = p_index[0] * self.w + p_index[1] + alpha
        return q

    # used to track how disparity changes after each expansion iteration
    def save_disparity_map(self):
        f = self.get_f_from_assignment_table()
        f = f * 9
        # l = 255.0 / (len(self.labels) + 1)
        # s = l
        # for label in self.labels:
        #     f[f==label] = s
        #     s += l

        cv2.imwrite("output/disparity.png", f)

    def initialize_assignment_function(self):
        a_table = np.random.randint(low=self.d_low, high=self.d_high, size=self.n)
        return a_table

    def get_f_from_assignment_table(self):
        f = self.assignment_table.reshape((self.h, self.w))
        return f

    def get_labeling_from_partition(self, G, label, A_0, A_alpha):
        # todo
        A_prime = np.zeros(self.n, dtype=np.int)
        for p in range(self.n):
            is_marked = False
            new_label = 0
            a1 = A_0[p]
            a2 = A_alpha[p]
            # marked as 1 if connected to 't' and 0 if connected to 's'
            a1_segment = G.get_segment(a1)
            if a1_segment == 0:
                is_marked = True
                new_label = self.assignment_table[p]
            a2_segment = G.get_segment(a2)
            if a2_segment == 1:
                if is_marked is not False:
                    print("override")
                new_label = label
            A_prime[p] = new_label

        return A_prime

    def calculate_expansion_tables(self, alpha):
        A_0 = np.zeros(self.n, dtype=np.int)
        vertex_label = 0
        pixel = 0
        for i in range(self.n):
            label = self.assignment_table[pixel]
            if label != 0 and label != alpha:
                A_0[pixel] = vertex_label
                vertex_label += 1
            pixel += 1

        A_alpha = np.arange(self.n)
        A_alpha += vertex_label
        return A_0, A_alpha, vertex_label + self.n

    def construct_graph(self, alpha, A_0, A_alpha, n):

        # todo: initialize with better params
        G = maxflow.Graph[int](2, 2)

        assignment_edges = {}

        G.add_nodes(n)
        print("Adding A_0 assignments")
        for p in range(self.n):
            a = A_0[p]
            if a == 0 and p != 0:
                # inactive assignment or assignment in A_alpha
                continue
            s_a_cost = self.get_s_a_cost(p, alpha, A_0)
            a_t_cost = self.get_a_t_cost(p, alpha, A_0, A_alpha)
            G.add_tedge(a, s_a_cost, a_t_cost)
            G = self.add_neighborhood_edges(p, G, assignment_edges, A_0, A_alpha)

        print("Adding A_alpha assignments")
        for p in range(self.n):
            a = A_0[p]
            if a == 0 and p != 0:
                # inactive assignment or assignment in A_alpha
                continue
            s_a_cost = self.get_s_a_cost(p, alpha, A_0)
            a_t_cost = self.get_a_t_cost(p, alpha, A_0, A_alpha)
            G.add_tedge(a, s_a_cost, a_t_cost)
            # an assignment a'=<p',q'> is a neighbor to another assignment a=<p,q> if
            # p is as neighbor of p' or q is a neighbor of q'
            G = self.add_neighborhood_edges(p, G, assignment_edges, A_0, A_alpha)
            G = self.add_expansion_edges(p, G, A_0, A_alpha)

        return G

    def D_a(self, p, d):
        # find the best match within the label range, clipped at thresh
        THRESHOLD = self.d_thresh
        p_index = np.unravel_index(p, (self.h, self.w))
        q_index = (p_index[0], p_index[1] + d)

        if q_index[1] > self.w - 1:
            return THRESHOLD

        # print(q_index)
        L_I_p = self.L[p_index[0], p_index[1], :]
        R_I_p = self.R[q_index[0], q_index[1], :]
        diff = L_I_p - R_I_p
        squared = diff ** 2
        value = np.sqrt(np.sum(squared))
        if value > THRESHOLD:
            return THRESHOLD
        return value

    def D_occ_p(self, p, A_0):
        if A_0[p] == 0:
            return self.lambda_v
        return 0.0

    def D_occ_a(self, p, q, A_0):
        return self.D_occ_p(p, A_0) + self.D_occ_p(q, A_0)

    def get_s_a_cost(self, p, alpha, A_0):
        q = self.get_q(p, alpha)
        if A_0[p] == 1:
            return self.D_occ_a(p, q, A_0)

        return self.D_a(p, q)

    def get_a_t_cost(self, p, alpha, A_0, A_alpha):
        q = self.get_q(p, alpha)
        if A_alpha[p] == 1:
            return self.D_occ_a(p, q, A_0)

        return self.D_a(p, q) + self.D_smooth(p, q, A_0, A_alpha)

    def D_smooth(self, p, q, A_0, A_alpha):
        # todo
        return 0.0

    def V(self):
        # todo
        return 0.0

    def add_neighborhood_edges(self, p, G, assignment_edges, A_0, A_alpha):
        # todo
        # an assignment a'=<p',q'> is a neighbor to another assignment a=<p,q> if
        # p is as neighbor of p' or q is a neighbor of q'

        # iterate through neighbors of p
        # if q < self.n + self.d_high - 1 and p < self.n - 1:
        #     # <q_2,p_2> has same disparity of <q,p>
        #     q_2 = q + 1
        #     p_2 = p + 1
            # is_assigned = self.assignment_table[q_2, p_2]
            # if assignment is different, add penalty

        return G

    def add_expansion_edges(self, p, G, A_0, A_alpha):

        if A_0[p] == 0:
            return G

        a1 = A_0[p]
        a2 = A_alpha[p]
        try:
            G.add_edge(a1, a2, 10000000, self.lambda_v)
        except Exception:
            print()

        return G

    def alpha_expansion(self, alpha, A_0, A_alpha, n):

        G = self.construct_graph(alpha, A_0, A_alpha, n)
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
            # n = number of assignment nodes for graph
            A_0, A_alpha, n = self.calculate_expansion_tables(label)
            cut_value, G = self.alpha_expansion(label, A_0, A_alpha, n)
            A_prime = self.get_labeling_from_partition(G, label, A_0, A_alpha)
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
        occluded_pixels = float(np.where(a_table == 0)[0].shape[0])

        return occluded_pixels * self.lambda_v

    def calculate_data_energy(self, a_table):
        sum = 0.0
        assignment_indices = np.where(a_table)
        for p in range(self.n):
            d = a_table[p]
            sum += self.D_a(p, d)

        return sum

    def calculate_smoothness_energy(self, f):
        pixel = 0
        sum = 0.0
        # todo

        return sum

