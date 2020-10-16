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
        self.num_particles = kwargs.get('num_particles') + 400 # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.template = template
        self.frame = frame
        h = frame.shape[0]
        w = frame.shape[1]
        # Initialize your particles array. Read the docstring. (x, y)
        self.particles = np.random.uniform(size=(self.num_particles, 2))
        self.particles[:, 0] = self.particles[:, 0] * w
        self.particles[:, 1] = self.particles[:, 1] * h

        # Initialize your weights array. Read the docstring.
        self.weights = np.ones(self.num_particles, dtype=np.float) / self.num_particles
        # Initialize any other components you may need when designing your filter.
        t_h = template.shape[0]
        t_w = template.shape[1]
        self.t_h_2 = int(t_h/2)
        self.t_w_2 = int(t_w/2)
        self.motion_variance = 14
        self.z = np.ones(self.num_particles, dtype=np.float) / self.num_particles
        # todo initialize beliefs?

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
        m = frame_cutout.shape[0]
        n = frame_cutout.shape[1]
        if template.shape[0] != frame_cutout.shape[0] or template.shape[1] != frame_cutout.shape[1]:
            raise Exception
        diff = template - frame_cutout
        squared = diff ** 2
        sum = np.sum(squared) / (m * n)
        return sum

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

        particle_samples = np.zeros((n, 2), dtype=np.float)
        cyclic_weights_index = 0
        for i in range(n):
            sample_generator = sorted_sample_generators[i]
            while sample_generator > cyclic_weights[cyclic_weights_index]:
                cyclic_weights_index += 1
            particle_samples[i] = self.particles[cyclic_weights_index]

        return particle_samples

    def update_weights(self):
        self.weights = self.z / np.sum(self.z)

    def update_measurement_likelihood(self, frame):
        # pad the frame border
        h = frame.shape[0]
        w = frame.shape[1]
        frame_copy = np.copy(frame)
        expanded_frame = cv2.copyMakeBorder(frame_copy, self.t_h_2, self.t_h_2, self.t_w_2, self.t_w_2, cv2.BORDER_REFLECT)
        mse = np.zeros_like(self.z)
        zeros = np.zeros_like(self.z)
        for i in range(self.particles.shape[0]):
            x, y = self.particles[i]
            x = int(x)
            y = int(y)
            if x < 0 or x >= w or y < 0 or y >= h:
                zeros[i] = 1
                continue
            frame_cutout = expanded_frame[y:y+2*self.t_h_2, x:x+2*self.t_w_2]
            mse[i] = self.get_error_metric(self.template, frame_cutout)

        # self.z = np.exp(mse/(-2*np.std(mse)))
        self.z = np.exp(mse/(-20))
        print(np.std(mse), self.z.min(), self.z.max())
        # if particle is out of frame, set probability to zero
        self.z[zeros == 1] = 0.0

    def diffuse(self):
        dynamics_matrix = np.random.uniform(low=-1*self.motion_variance, high=self.motion_variance, size=(self.num_particles, 2))
        self.particles = self.particles + dynamics_matrix

    def get_mean(self):
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        x_w = np.multiply(x, self.weights)
        y_w = np.multiply(y, self.weights)

        return np.sum(x_w), np.sum(y_w)

    def render_best_weights(self, frame_in):

        avg = np.average(self.weights)
        std = np.std(self.weights)
        q = np.where(self.weights > avg + std/2)

        n = q[0].shape[0]

        # best_particles = np.zeros((n, 2))
        best_particles = self.particles[q]

        for i in range(n):
            y, x = best_particles[i]
            x = int(x)
            y = int(y)
            cv2.circle(frame_in, (y, x), 2, (0, 0, 255), 2)

        cv2.imwrite("best.png", frame_in)

    def show_error_function(self, frame):

        h = frame.shape[0]
        w = frame.shape[1]
        mse_m = np.zeros((h, w), dtype=np.float)
        expanded_frame = cv2.copyMakeBorder(frame, self.t_h_2, self.t_h_2, self.t_w_2, self.t_w_2, cv2.BORDER_REFLECT)

        for i in range(h):
            for j in range(w):
                frame_cutout = expanded_frame[i:i+2*self.t_w_2, j:j+2*self.t_h_2]
                mse_m[i, j] = self.get_error_metric(self.template, frame_cutout)

        u = np.exp(mse_m/(-2*np.std(mse_m)))
        u = u / u.max() * 255
        cv2.imwrite("out/error_map.png", u)
        pass

    def process(self, frame, i=0):
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
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # move particles by gaussian noise
        self.render(np.copy(frame), "out/frame_0" + str(i) + ".png")

        # self.show_error_function(np.copy(frame))
        self.render(np.copy(frame), "out/frame_1" + str(i) + ".png")
        # likelihood of measurement
        self.update_measurement_likelihood(frame)
        # update weights
        self.update_weights()
        self.render_best_weights(frame)
        # resample
        sampled_particles = self.resample_particles()
        self.particles = sampled_particles
        self.render(np.copy(frame), "out/frame_2" + str(i) + ".png")
        self.diffuse()
        self.render(np.copy(frame), "out/frame_3" + str(i) + ".png")

        # get mean x, mean y etc
        x, y = self.get_mean()
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

        for i in range(self.num_particles):
            x, y = self.particles[i]
            x = int(x)
            y = int(y)
            cv2.circle(frame_in, (x, y), 2, (0, 0, 255), 2)
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        if file is not None and render:
            cv2.imwrite(file, frame_in)
        # raise NotImplementedError


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

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

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
        raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

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
        raise NotImplementedError