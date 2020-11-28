import numpy as np
import maxflow
import cv2


class AlphaExpansion:

    def __init__(self, left, right, labels, lambda_v, d_thresh, K, full_n, reverse):
        self.L = left.astype(np.float)
        self.R = right.astype(np.float)
        self.labels = labels
        self.h = left.shape[0]
        self.w = left.shape[1]
        self.n = self.h * self.w
        self.d_low = labels[0]
        self.d_high = labels[labels.shape[0]-1]
        self.assignment_table, self.q_assignment_table = self.initialize_assignment_function()
        self.lambda_v = lambda_v
        self.d_thresh = d_thresh
        # variable to make duplicate edges are not added for neighboring assignments <a_1, a_2>
        self.assignment_edges = {}
        self.neighbors = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        self.iteration = 1
        self.K = K
        self.reverse = reverse
        if full_n:
            self.neighbors = np.array([[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,1], [1, -1], [-1,-1]])

    # returns the assigned pixel to p
    def get_q(self, p, alpha):
        p_index = np.unravel_index(p, (self.h, self.w))
        if p_index[1] + alpha > self.w - 1 or p_index[1] + alpha < 0:
            return p

        q = p_index[0] * self.w + p_index[1] + alpha
        return q

    def get_p(self, q, alpha):
        q_index = np.unravel_index(q, (self.h, self.w))
        if q_index[1] + alpha > self.w - 1 or q_index[1] + alpha < 0:
            return q

        q = q_index[0] * self.w + q_index[1] - alpha
        return q

    def get_index(self, i, j):
        return i * self.w + j

    # used to track how disparity changes after each expansion iteration
    def save_disparity_map(self):
        f = self.get_f_from_assignment_table()
        f = f * 9
        if self.reverse:
            f *= -1

        cv2.imwrite("output/disparity/d_" + str(self.iteration) + ".png", f)

    def initialize_assignment_function(self):
        high = np.max(self.labels)
        low = np.min(self.labels)
        a_table = np.random.randint(low=low, high=high, size=self.n)
        q_table = np.zeros_like(a_table)
        for p in range(self.n):
            p_index = np.unravel_index(p, (self.h, self.w))
            d = a_table[p]
            q_index = (p_index[0], p_index[1] + d)
            if q_index[1] > self.w - 1:
                continue
            q = self.get_index(q_index[0], q_index[1])
            q_table[q] = d
        return a_table, q_table

    def get_f_from_assignment_table(self):
        f = self.assignment_table.reshape((self.h, self.w))
        return f

    def get_labeling_from_partition(self, G, label, A_0, A_alpha):
        A_prime = np.zeros(self.n, dtype=np.int)
        # todo
        Q_prime = np.zeros(self.n, dtype=np.int)
        for p in range(self.n):
            is_marked = False
            # label = self.assignment_table[]
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
            # set q_table
            p_index = np.unravel_index(p, (self.h, self.w))
            q_index = (p_index[0], p_index[1] + new_label)
            if q_index[1] > self.w - 1 or q_index[1] < 0:
                continue
            q = self.get_index(q_index[0], q_index[1])
            Q_prime[q] = new_label

        return A_prime, Q_prime

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
        G = maxflow.Graph[int](self.n, self.n)

        self.assignment_edges = {}

        G.add_nodes(n)
        print("Adding A_0 assignments")
        for p in range(self.n):
            a_0 = A_0[p]
            G = self.add_neighborhood_edges(p, G, A_alpha, alpha)
            a_alpha = A_alpha[p]
            q_alpha = self.get_q(p, alpha)
            d_a_alpha_occ = self.D_occ_a(p, q_alpha, A_0)
            d_a_alpha = self.D_a(p, alpha)
            G.add_tedge(a_alpha, d_a_alpha, d_a_alpha_occ)

            if a_0 != 0 or p != 0:
                label = self.assignment_table[p]
                # G = self.add_neighborhood_edges(p, G, A_0, label)
                q_0 = self.get_q(p, label)
                d_a_0_occ = self.D_occ_a(p, q_0, A_0)
                d_smooth = self.D_smooth(p, self.assignment_table, self.q_assignment_table, alpha)
                d_a_0 = self.D_a(p, label) + d_smooth
                G.add_tedge(a_0, d_a_0_occ, d_a_0)
                G.add_edge(a_0, a_alpha, 10000000, self.K * self.lambda_v)
                # print(p, d_a_alpha, d_a_alpha_occ, d_a_0_occ, d_a_0)

        return G

    def D_a(self, p, d):
        # find the best match within the label range, clipped at thresh
        THRESHOLD = self.d_thresh
        p_index = np.unravel_index(p, (self.h, self.w))
        q_index = (p_index[0], p_index[1] + d)

        if q_index[1] > self.w - 1 or q_index[1] < 0:
            return THRESHOLD

        # print(q_index)
        L_I_p = self.L[p_index[0], p_index[1], :]
        R_I_p = self.R[q_index[0], q_index[1], :]
        # L_I_p = self.L[p_index[0], p_index[1]]
        # R_I_p = self.R[q_index[0], q_index[1]]
        diff = L_I_p - R_I_p
        squared = diff ** 2
        value = np.sqrt(np.sum(squared))
        if value > THRESHOLD:
            return THRESHOLD
        return value

    def D_occ_p(self, p, A_0):
        # return self.lambda_v
        # if A_0[p] == 0:
        #     return 0.0
        return self.K * self.lambda_v

    def D_occ_a(self, p, q, A_0):
        return self.D_occ_p(p, A_0) + self.D_occ_p(q, A_0)

    def V(self, a_1_label, a_2_label, a_1, a_2):
        if a_1_label == a_2_label:
            return 0.0
        p, q = a_1
        r, s = a_2
        p_r_sum = 0
        q_s_sum = 0

        if 0 <= p <= self.w*self.h - 1 and 0 <= r <= self.w*self.h - 1:
            p_index = np.unravel_index(p, (self.h, self.w))
            r_index = np.unravel_index(r, (self.h, self.w))
            I_p = self.L[p_index[0], p_index[1]]
            I_r = self.L[r_index[0], r_index[1]]
            p_r_sum = np.sum(np.abs(I_p - I_r))

        if 0 <= q <= self.w*self.h - 1 and 0 <= s <= self.w*self.h - 1:
            q_index = np.unravel_index(q, (self.h, self.w))
            s_index = np.unravel_index(s, (self.h, self.w))
            I_q = self.L[q_index[0], q_index[1]]
            I_s = self.L[s_index[0], s_index[1]]
            q_s_sum = np.sum(np.abs(I_q - I_s))

        value = np.max([p_r_sum, q_s_sum])
        if value < 8:
            return 3 * self.lambda_v

        return self.lambda_v

    def add_neighborhood_edges(self, p, G, A_alpha, alpha):
        p_index = np.unravel_index(p, (self.h, self.w))
        self.add_n_edges(G, p_index, A_alpha, alpha)
        return G

    def add_n_edges(self, G, p_index, A, alpha):
        p = self.get_index(p_index[0], p_index[1])
        q = self.get_q(p, alpha)

        a_1 = A[p]
        # p_prime is close to p or q_prime is close to q
        for i in range(self.neighbors.shape[0]):
            p_prime_index = p_index + self.neighbors[i]
            if np.min(p_prime_index) < 0:
                continue
            if p_prime_index[1] > self.w - 1 or p_prime_index[0] > self.h - 1:
                continue
            p_prime = self.get_index(p_prime_index[0], p_prime_index[1])
            q_prime = self.get_q(p_prime, alpha)
            p_prime_label = self.assignment_table[p_prime]

            if p_prime_label == alpha:
                continue
            # check if labels match
            v = self.V(p_prime_label, alpha, (p, q), (p_prime, q_prime))
            a_2 = A[p_prime]
            G.add_edge(a_1, a_2, v, 0)
            # if not self.is_edge_present(a_1, a_2):
            #     G.add_edge(a_1, a_2, v, v)

        # q = self.get_q(p, alpha)
        q_index = np.unravel_index(q, (self.h, self.w))
        a_1 = A[q]
        q = self.get_index(p_index[0], p_index[1])
        for i in range(self.neighbors.shape[0]):
            q_prime_index = q_index + self.neighbors[i]
            if np.min(q_prime_index) < 0:
                continue
            if q_prime_index[1] > self.w - 1 or q_prime_index[0] > self.h - 1:
                continue
            q_prime = self.get_index(q_prime_index[0], q_prime_index[1])
            p_prime = self.get_p(q_prime, alpha)
            q_prime_label = self.q_assignment_table[q_prime]
            if q_prime_label == alpha:
                continue
            # check if labels match
            v = self.V(q_prime_label, alpha, (p, q), (p_prime, q_prime))
            a_2 = A[q_prime]
            G.add_edge(a_1, a_2, v, 0)
            # if not self.is_edge_present(a_1, a_2):
            #     G.add_edge(a_1, a_2, v, v)

        return G

    def is_edge_present(self, a_1, a_2):
        if a_1 > a_2:
            v = self.assignment_edges.get(a_1)
            if v is None:
                new_v = {a_2}
                self.assignment_edges[a_1] = new_v
                return True
            else:
                if a_2 not in v:
                    # add a_2 to v
                    v.add(a_2)
                    self.assignment_edges[a_1] = v
                    return True
                else:
                    return False
        else:
            v = self.assignment_edges.get(a_2)
            if v is None:
                new_v = {a_1}
                self.assignment_edges[a_2] = new_v
                return True
            else:
                if a_1 not in v:
                    # add a_2 to v
                    v.add(a_1)
                    self.assignment_edges[a_2] = v
                    return True
                else:
                    return False

    def alpha_expansion(self, alpha, A_0, A_alpha, n):

        G = self.construct_graph(alpha, A_0, A_alpha, n)
        flow = G.maxflow()
        return flow, G

    def calculate_alpha_expansion(self):
        self.save_disparity_map()
        current_energy = self.calculate_energy(self.assignment_table, self.q_assignment_table)
        print("current energy", current_energy)

        labels = self.labels

        arr = np.arange(labels.shape[0])
        np.random.shuffle(arr)
        for i in arr:
            label = labels[i]
            # n = number of assignment nodes for graph
            A_0, A_alpha, n = self.calculate_expansion_tables(label)
            cut_value, G = self.alpha_expansion(label, A_0, A_alpha, n)
            A_prime, Q_prime = self.get_labeling_from_partition(G, label, A_0, A_alpha)
            energy_after_expansion = self.calculate_energy(A_prime, Q_prime)
            print("energy after expansion", energy_after_expansion, label)
            if energy_after_expansion < current_energy:
                return True, A_prime, Q_prime

        return False, None, None

    def calculate_disparity_map(self):

        has_expansion_reduced_energy = True

        while has_expansion_reduced_energy:
            has_expansion_reduced_energy, A_prime, Q_prime = self.calculate_alpha_expansion()
            if has_expansion_reduced_energy:
                self.assignment_table = A_prime
                self.q_assignment_table = Q_prime
            self.iteration += 1

        f = self.get_f_from_assignment_table()
        return f

    """
    ------------------------------ energy functions
    """
    # only for A_0
    def D_smooth(self, p, A_table, Q_table, alpha):
        # todo
        a_1_label = A_table[p]
        p_index = np.unravel_index(p, (self.h, self.w))
        q = self.get_q(p, a_1_label)
        value = 0.0

        for i in range(self.neighbors.shape[0]):
            p_prime_index = p_index + self.neighbors[i]
            if np.min(p_prime_index) < 0:
                continue
            if p_prime_index[1] > self.w - 1 or p_prime_index[0] > self.h - 1:
                continue
            p_prime = self.get_index(p_prime_index[0], p_prime_index[1])
            q_prime = self.get_q(p_prime, alpha)
            a_2_label = A_table[p_prime]
            # if a_2_label == alpha:
            #     continue
            value += self.V(a_1_label, a_2_label, (p,q), (p_prime, q_prime))

        q_index = np.unravel_index(q, (self.h, self.w))
        a_1_label = Q_table[q]

        for i in range(self.neighbors.shape[0]):
            q_prime_index = q_index + self.neighbors[i]
            if np.min(q_prime_index) < 0:
                continue
            if q_prime_index[1] > self.w - 1 or q_prime_index[0] > self.h - 1:
                continue
            q_prime = self.get_index(q_prime_index[0], q_prime_index[1])
            p_prime = self.get_p(q_prime, alpha)
            a_2_label = Q_table[q_prime]
            # if a_2_label == alpha:
            #     continue
            value += self.V(a_1_label, a_2_label, (p,q), (p_prime, q_prime))

        return value

    def calculate_energy(self, a_table, q_table):
        smooth = self.calculate_smoothness_energy(a_table, q_table)
        data = self.calculate_data_energy(a_table)
        occ = self.calculate_occlusion_energy(a_table)

        return data + smooth + occ

    def calculate_occlusion_energy(self, a_table):
        occluded_pixels = float(np.where(a_table == 0)[0].shape[0])

        return occluded_pixels * self.K * self.lambda_v

    def calculate_data_energy(self, a_table):
        sum = 0.0
        for p in range(self.n):
            d = a_table[p]
            sum += self.D_a(p, d)

        return sum

    def calculate_smoothness_energy(self, a_table, q_table):
        sum = 0.0
        for p in range(self.n):
            alpha = -1
            sum += self.D_smooth(p, a_table, q_table, alpha)

        return sum


