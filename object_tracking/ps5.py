"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.measurement_noise = R
        self.process_noise = Q
        self.covariance_matrix = np.eye(4) * 0.1
        # measurement matrix
        self.M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        # transition matrix
        self.D = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def predict(self):
        self.state = np.matmul(self.D, self.state)
        temp = np.matmul(self.D, self.covariance_matrix)
        self.covariance_matrix = np.matmul(temp, self.D.T) + self.process_noise

    def correct(self, meas_x, meas_y):
        # create measurement vector
        z = np.zeros((2, 1), dtype=np.float)
        z[0, 0] = meas_x
        z[1, 0] = meas_y

        # calculate kalman gain
        temp = np.matmul(self.M, self.covariance_matrix)
        inverse = np.linalg.inv(np.matmul(temp, self.M.T) + self.measurement_noise)
        kalman_gain = np.matmul(self.covariance_matrix, np.matmul(self.M.T, inverse))

        # update state
        self.state = self.state + np.squeeze(np.matmul(kalman_gain, z - np.expand_dims(np.matmul(self.M, self.state), axis=1)))
        # update covariance
        self.covariance_matrix = np.matmul(np.eye(4) - np.matmul(kalman_gain, self.M), self.covariance_matrix)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles') + 1700 # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp') + 15 # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')   # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        # template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.use_threshold = kwargs.get('use_threshold', False)
        self.threshold = kwargs.get('threshold', 42)
        self.sigma_exp_original = self.sigma_exp
        self.sigma_dyn_original = self.sigma_dyn
        self.template = template
        self.in_gray_mode = kwargs.get('in_gray_mode', True)
        if self.in_gray_mode:
            self.template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY).astype(np.float)

        # self.original_template = template
        self.frame = frame
        h = frame.shape[0]
        w = frame.shape[1]
        self.h = h
        self.w = w
        # Initialize your particles array. Read the docstring. (x, y)
        self.particles = np.random.uniform(size=(self.num_particles, 2))
        use_box_initialization = kwargs.get('use_box_initialization', False)
        box = kwargs.get('box', {})
        if use_box_initialization:
            self.particles[:, 0] = np.random.uniform(low=box.get("x_min"), high=box.get("x_max"), size=self.num_particles)
            self.particles[:, 1] = np.random.uniform(low=box.get("y_min"), high=box.get("y_max"), size=self.num_particles)
        else:
            self.particles[:, 0] = self.particles[:, 0] * w
            self.particles[:, 1] = self.particles[:, 1] * h
        # self.particles[:, 2] = 1.0

        # Initialize your weights array. Read the docstring.
        # self.in_template_adjust_mode = False
        self.in_template_adjust_mode = False
        self.weights = np.ones(self.num_particles, dtype=np.float) / self.num_particles
        # Initialize any other components you may need when designing your filter.
        t_h = template.shape[0]
        t_w = template.shape[1]
        self.t_h = t_h
        self.t_w = t_w
        self.t_h_2 = int(t_h/2)
        self.t_w_2 = int(t_w/2)
        self.z = np.ones(self.num_particles, dtype=np.float) / self.num_particles
        self.time = 0

    def get_particle(self, i):
        if self.particles.shape[1] == 3:
            x, y, t_size = self.particles[i]
            x = int(x)
            y = int(y)
            return x, y, t_size
        x, y = self.particles[i]
        x = int(x)
        y = int(y)
        return x, y

    def increment_time(self):
        self.time += 1

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        template = template.astype(np.float)
        frame_cutout = frame_cutout.astype(np.float)
        m = frame_cutout.shape[0]
        n = frame_cutout.shape[1]
        if template.shape[0] != frame_cutout.shape[0] or template.shape[1] != frame_cutout.shape[1]:
            raise Exception

        if self.in_gray_mode:
            diff = template - frame_cutout
            squared = diff ** 2
            return np.sum(squared) / (m * n)

        mse = 0.0
        for i in range(3):
            diff = template[:,:,i] - frame_cutout[:,:,i]
            squared = diff ** 2
            mse += np.sum(squared) / (m * n)

        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        n = self.weights.shape[0]
        cyclic_weights = np.zeros(n, dtype=np.float)
        for i in range(n-1):
            cyclic_weights[i+1] = cyclic_weights[i] + self.weights[i]

        sample_generators = np.random.uniform(size=self.num_particles) * cyclic_weights.max()

        sorted_sample_generators = np.sort(sample_generators)

        particle_samples = np.zeros_like(self.particles, dtype=np.float)
        weights = np.zeros_like(self.weights)
        cyclic_weights_index = 0
        for i in range(n):
            sample_generator = sorted_sample_generators[i]
            while sample_generator > cyclic_weights[cyclic_weights_index]:
                cyclic_weights_index += 1
            particle_samples[i] = self.particles[cyclic_weights_index]
            weights[i] = self.weights[cyclic_weights_index]

        return particle_samples, weights

    def update_weights(self):
        self.weights *= self.z / np.sum(self.z)
        self.weights = self.weights / np.sum(self.weights)

    def expand_frame(self, frame, adjusted_template):
        t_h_2 = int(adjusted_template.shape[0] / 2)
        t_w_2 = int(adjusted_template.shape[1] / 2)
        frame_copy = np.copy(frame)
        expanded_frame = cv2.copyMakeBorder(frame_copy, t_h_2, t_h_2, t_w_2, t_w_2, cv2.BORDER_REFLECT)
        return expanded_frame

    def get_adjusted_template(self, template_size):
        h = int(self.t_h * template_size)
        w = int(self.t_w * template_size)
        adjusted_template = cv2.resize(np.copy(self.template), (w, h))
        return adjusted_template

    def get_cutout_bounds(self, adjusted_template):
        y_bound = adjusted_template.shape[0]
        x_bound = adjusted_template.shape[1]
        return x_bound, y_bound

    def calculate_mse_error(self, frame):
        h = frame.shape[0]
        w = frame.shape[1]

        adjusted_template = np.copy(self.template)
        expanded_frame = self.expand_frame(frame, adjusted_template)

        mse = np.ones_like(self.z)
        zeros = np.zeros_like(self.z)
        for i in range(self.particles.shape[0]):
            particle = self.get_particle(i)
            if len(particle) == 2:
                x, y = particle
                t_size = 1.0
            else:
                x, y, t_size = particle
            if x < 0 or x >= w or y < 0 or y >= h:
                zeros[i] = 1
            if self.in_template_adjust_mode:
                adjusted_template = self.get_adjusted_template(t_size)
                expanded_frame = self.expand_frame(frame, adjusted_template)

            x_bound, y_bound = self.get_cutout_bounds(adjusted_template)
            frame_cutout = expanded_frame[y:y+y_bound, x:x+x_bound]
            mse[i] = self.get_error_metric(adjusted_template, frame_cutout)
        return mse, zeros

    def update_measurement_likelihood(self, frame):
        mse, zeros = self.calculate_mse_error(frame)

        self.z = np.exp(mse/(-2.0*self.sigma_exp*self.sigma_exp))
        self.z[zeros == 1] = 0.0
        self.z = self.z / np.sum(self.z)

    def round_particles(self):
        self.particles[self.particles < 0] = 0
        x = self.particles[:,0]
        y = self.particles[:,1]
        x[x > self.w-1] = self.w-1
        y[y > self.h-1] = self.h-1
        self.particles[:,0] = x
        self.particles[:,1] = y

    def diffuse(self):
        dynamics_matrix = np.random.normal(0.0, self.sigma_dyn, size=(self.num_particles, 2))

        self.particles[:, 0] += dynamics_matrix[:, 0]
        self.particles[:, 1] += dynamics_matrix[:, 1]
        self.round_particles()

    def get_mean(self):
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        t_avg = 1.0
        if self.particles.shape[1] == 3:
            t_size = self.particles[:, 2]
            t_avg = np.multiply(t_size, self.weights)
        x_w = np.multiply(x, self.weights)
        y_w = np.multiply(y, self.weights)

        return np.sum(x_w), np.sum(y_w), np.sum(t_avg)

    # def render_best_weights(self, frame_in):
    #
    #     avg = np.average(self.weights)
    #     std = np.std(self.weights)
    #     q = np.where(self.weights > avg + std/2)
    #
    #     n = q[0].shape[0]
    #
    #     best_particles = self.particles[q]
    #
    #     for i in range(n):
    #         y, x = best_particles[i]
    #         x = int(x)
    #         y = int(y)
    #         cv2.circle(frame_in, (y, x), 2, (0, 0, 255), 2)
    #
    #     cv2.imwrite("best.png", frame_in)
    #
    # def show_error_function(self, frame):
    #     cv2.imwrite("out/t_" + str(self.time) + ".png", self.template)
    #
    #     h = frame.shape[0]
    #     w = frame.shape[1]
    #     mse_m = np.zeros((h, w), dtype=np.float)
    #     mse_m[:,:] = 10000
    #     adjusted_template = np.copy(self.template)
    #     expanded_frame = self.expand_frame(frame, adjusted_template)
    #
    #     for i in range(0, h):
    #         for j in range(0, w):
    #             x_bound, y_bound = self.get_cutout_bounds(adjusted_template)
    #             frame_cutout = expanded_frame[i:i+y_bound, j:j+x_bound]
    #             mse_m[i, j] = self.get_error_metric(adjusted_template, frame_cutout)
    #
    #     # u = np.exp(mse_m/(-2*np.std(mse_m)))
    #     u = np.exp(mse_m/(-15000))
    #
    #     # u[mse_m == 0] = 0
    #     u = u / u.max() * 255
    #     cv2.imwrite("out/error_map.png", u)
    #     pass

    def particle_distibution(self):
        # x_min = self.particles[:,0].min()
        # x_max = self.particles[:,0].max()
        # y_min = self.particles[:,1].min()
        # y_max = self.particles[:,1].max()
        x_std = np.std(self.particles[:, 0])
        y_std = np.std(self.particles[:, 1])
        # print(x_min, x_max, x_std, " --- ", y_min, y_max, y_std, self.min_mse, self.current_mse)
        return x_std, y_std

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.increment_time()
        if self.in_gray_mode and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float)
        # self.particle_distibution()
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # move particles by gaussian noise
        self.render(np.copy(frame), "out/frames/frame_" + str(self.time) + "_0.png")
        # self.show_error_function(np.copy(frame))
        # likelihood of measurement
        self.update_measurement_likelihood(frame)
        # update weights
        self.update_weights()
        # self.render_best_weights(frame)
        # resample
        sampled_particles, weights = self.resample_particles()
        self.particles = sampled_particles
        self.weights = weights
        self.weights = self.weights / np.sum(self.weights)

        self.render(np.copy(frame), "out/frame_" + str(self.time) + "_1.png")

        self.diffuse()
        self.render(np.copy(frame), "out/frame_" + str(self.time) + "_2.png")

        # get mean x, mean y etc
        x, y, _ = self.get_mean()
        return x, y

    def render(self, frame_in, file=None, render=False):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0
        size_weighted_mean = 0

        for i in range(self.num_particles):
            particle = self.get_particle(i)
            if len(particle) == 2:
                x, y = particle
                t_size = 1.0
            else:
                x, y, t_size = particle
            # x, y, size = self.get_particle(i)
            cv2.circle(frame_in, (x, y), 1, (0, 0, 255), 1)
            x_weighted_mean += x * self.weights[i]
            y_weighted_mean += y * self.weights[i]
            size_weighted_mean += t_size * self.weights[i]
            # x_weighted_mean += self.particles[i, 0] * self.weights[i]
            # y_weighted_mean += self.particles[i, 1] * self.weights[i]
            # size_weighted_mean += self.particles[i, 2] * self.weights[i]

        h_2 = self.t_h_2 * size_weighted_mean
        w_2 = self.t_w_2 * size_weighted_mean
        left_x = int(x_weighted_mean - w_2)
        right_x = int(x_weighted_mean + w_2)

        top_y = int(y_weighted_mean - h_2)
        bottom_y = int(y_weighted_mean + h_2)

        cv2.rectangle(frame_in,(left_x, top_y),(right_x, bottom_y),(0,255,0),3)

        x_d = self.particles[:, 0] - x_weighted_mean
        y_d = self.particles[:, 1] - y_weighted_mean
        z_d = np.sqrt(x_d ** 2 + y_d ** 2)
        z_d_weighted = np.multiply(z_d, self.weights)
        z_d = np.sum(z_d_weighted)
        if z_d < 200:
            cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(z_d), (255, 0, 0), 2)

        # Complete the rest of the code as instructed.
        # if file is not None and render:
        #     cv2.imwrite(file, frame_in)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.sigma_exp += -5
        self.sigma_dyn -= 0
        self.in_template_adjust_mode = False
        self.alpha = kwargs.get('alpha') # + 0.1 # required by the autograder
        self.use_constant_alpha = kwargs.get('use_constant_alpha', True)
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def get_average_patch(self, frame):
        x_avg, y_avg, t_size_avg = self.get_mean()
        # print(t_size_avg)
        avg_patch = self.get_patch(int(x_avg), int(y_avg), frame, t_size_avg)
        return avg_patch

    def get_patch(self, x, y, frame, t_size):
        adjusted_template = self.get_adjusted_template(t_size)
        x_bound, y_bound = self.get_cutout_bounds(adjusted_template)
        expanded_frame = self.expand_frame(frame, adjusted_template)
        best_patch = expanded_frame[y:y+y_bound, x:x+x_bound]
        return best_patch

    def get_best_patch(self, frame):

        mse, _ = self.calculate_mse_error(frame)

        best_mse_index = np.argmin(mse)
        particle = self.get_particle(best_mse_index)
        if len(particle) == 2:
            x, y = particle
            t_size = 1.0
        else:
            x, y, t_size = particle
        # x, y, t_size = self.get_particle(best_mse_index)
        best_patch = self.get_patch(x, y, frame, t_size)
        return best_patch

    def get_template_error(self, t_1, t_2):

        if not self.in_gray_mode:
            t_1_gray = cv2.cvtColor(t_1.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float)
            t_2_gray = cv2.cvtColor(t_2,cv2.COLOR_BGR2GRAY).astype(np.float)
        else:
            t_1_gray = t_1
            t_2_gray = t_2

        diff = t_1_gray - t_2_gray
        squared = diff * diff
        error = np.sum(squared) / (256 * 256)
        return error

    def get_alpha(self):
        x_std, y_std = self.particle_distibution()
        alpha = np.exp(-1 * (x_std + y_std)/50)
        return alpha

    def update_template(self, frame, write=True):
        previous_template = np.copy(self.template)

        best_patch = self.get_best_patch(frame)
        h = best_patch.shape[0]
        w = best_patch.shape[1]
        resized_previous = cv2.resize(previous_template, (w,h))
        template_error = self.get_template_error(resized_previous, best_patch)
        if self.use_threshold:
            if template_error < self.threshold:
                self.sigma_dyn = self.sigma_dyn_original
                self.sigma_exp = self.sigma_exp_original
            else:
                self.sigma_dyn = self.sigma_dyn_original + 10
                self.sigma_exp = self.sigma_exp_original - 35
        if self.use_constant_alpha:
            alpha = self.alpha # 0.05
        else:
            alpha = self.get_alpha()
        # alpha = self.alpha
        # print(self.time, alpha, template_error)

        new_template = alpha * best_patch + (1 - alpha) * resized_previous
        # self.update_previous_templates(new_template, template_error)
        self.template = new_template
        if write:
            cv2.imwrite("out/best_template/t_" + str(self.time) + ".png", best_patch)
            cv2.imwrite("out/template/t_" + str(self.time) + ".png", self.template)
            cv2.imwrite("out/previous_template/t_" + str(self.time) + ".png", previous_template)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        if self.in_gray_mode:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # self.detect_occlusion(frame)

        self.update_template(frame, write=True)
        x, y = super(AppearanceModelPF, self).process(frame)
        return x, y
        # raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.in_template_adjust_mode = True
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        h = frame.shape[0]
        w = frame.shape[1]
        self.use_constant_alpha = kwargs.get('use_constant_alpha', True)
        self.min_d = kwargs.get('min_d', -0.015)
        self.max_d = kwargs.get('max_d', 0.015)
        self.use_alpha_blending = kwargs.get('use_alpha_blending', False)
        self.particles = np.random.uniform(size=(self.num_particles, 3))
        use_box_initialization = kwargs.get('use_box_initialization', False)
        box = kwargs.get('box', {})
        if use_box_initialization:
            self.particles[:, 0] = np.random.uniform(low=box.get("x_min"), high=box.get("x_max"), size=self.num_particles)
            self.particles[:, 1] = np.random.uniform(low=box.get("y_min"), high=box.get("y_max"), size=self.num_particles)
        else:
            self.particles[:, 0] = self.particles[:, 0] * w
            self.particles[:, 1] = self.particles[:, 1] * h
        self.particles[:, 2] = 1.0


    def diffuse_template_size(self):
        template_size_dynamics_vector = np.random.uniform(low=self.min_d, high=self.max_d, size=(self.num_particles))
        t_size = self.particles[:, 2] + template_size_dynamics_vector
        # don't let template get 30% smaller than original
        d = 0.3
        t_size[t_size < d] = d
        self.particles[:, 2] = t_size
        # don't let template get 250% larger than original
        d = 2.5
        t_size[t_size > d] = d
        self.particles[:, 2] = t_size

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        if self.use_alpha_blending:
            self.update_template(frame, write=True)
        x, y = super(AppearanceModelPF, self).process(frame)
        self.diffuse_template_size()
        # cv2.imwrite("out/template/t_" + str(self.time) + ".png", self.template)
        return x, y
