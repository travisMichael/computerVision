"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    n = size[0] * size[1]
    image_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    m = len(image_files)

    x = np.zeros((m, n))
    y = np.zeros(m)

    for i in range(len(image_files)):
        file = image_files[i]
        subject = int(file.split(".")[0].split('t')[1])
        image = cv2.imread(folder + "/" + file, 0)
        image_resized = cv2.resize(image, (size[0], size[1]))
        flattened = np.ndarray.flatten(image_resized)
        x[i,:] = flattened
        y[i] = subject

    return x, y


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    m = X.shape[0]
    indices = np.arange(m)
    random_indices = np.random.permutation(indices)
    n = int(m * p)

    X_train = X[random_indices[0:(m-n)], :]
    X_test = X[random_indices[(m-n):m], :]

    y_train = y[random_indices[0:(m-n)]]
    y_test = y[random_indices[(m-n):m]]

    return X_train, y_train, X_test, y_test


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    mean_f = np.mean(x, axis=0)
    return mean_f


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    n = X.shape[1]
    mean_f = get_mean_face(X)
    diff = X - mean_f
    exp_1 = np.expand_dims(diff, axis=2)
    exp_2 = np.expand_dims(diff, axis=1)

    temp = np.matmul(exp_1, exp_2)
    cov = np.sum(temp, axis=0)

    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    k_eigen_values = eigen_values[n-k:n]
    k_eigen_vectors = eigen_vectors[:, n-k:n]

    flipped_values = np.flip(k_eigen_values, axis=0)
    flipped_vectors = np.flip(k_eigen_vectors, axis=1)

    return flipped_vectors, flipped_values


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def normalize_weights(self):
        self.weights = self.weights / np.sum(self.weights)

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(self.num_iterations):
            self.normalize_weights()
            wk_clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            wk_clf.train()
            wk_results = [wk_clf.predict(x) for x in self.Xtrain]
            non_matching_indices = np.where(wk_results != self.ytrain)[0]
            e = np.sum(self.weights[non_matching_indices])
            a = np.log((1-e)/e) / 2.0
            self.alphas.append(a)
            self.weakClassifiers.append(wk_clf)
            if e < self.eps:
                break
            new_weights = np.exp(wk_results * self.ytrain * -1 * a)
            self.weights += new_weights

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        boosting_results = self.predict(self.Xtrain)
        matching_indices = np.where(boosting_results == self.ytrain)[0]
        correct = matching_indices.shape[0]
        total = self.ytrain.shape[0]
        incorrect = total - correct
        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        m = X.shape[0]
        summed_predictions = np.zeros(m, dtype=np.float)
        k = len(self.alphas)
        for i in range(k):
            wk_clf = self.weakClassifiers[i]
            a = self.alphas[i]
            predictions = [wk_clf.predict(x) for x in X]
            weighted_prediction = np.array(predictions, dtype=np.float) * a
            summed_predictions += weighted_prediction

        return np.sign(summed_predictions)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature = np.zeros((shape[0], shape[1]), dtype=np.float)
        embedded_feature = np.zeros((self.size[0], self.size[1]), dtype=np.float)
        mid_point = int(self.size[0] / 2)
        embedded_feature[0:mid_point, :] = 255.0
        embedded_feature[mid_point:self.size[0], :] = 126.0
        feature[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = embedded_feature
        return feature

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature = np.zeros((shape[0], shape[1]), dtype=np.float)
        embedded_feature = np.zeros((self.size[0], self.size[1]), dtype=np.float)
        mid_point = int(self.size[1] / 2)
        embedded_feature[:, 0:mid_point] = 255.0
        embedded_feature[:, mid_point:self.size[1]] = 126.0
        feature[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = embedded_feature
        return feature

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature = np.zeros((shape[0], shape[1]), dtype=np.float)
        embedded_feature = np.zeros((self.size[0], self.size[1]), dtype=np.float)
        embedded_feature[:, :] = 255.0
        boundary_point = int(self.size[0] / 3)
        embedded_feature[boundary_point:boundary_point*2, :] = 126.0
        feature[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = embedded_feature
        return feature

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature = np.zeros((shape[0], shape[1]), dtype=np.float)
        embedded_feature = np.zeros((self.size[0], self.size[1]), dtype=np.float)
        embedded_feature[:, :] = 255.0
        boundary_point = int(self.size[1] / 3)
        embedded_feature[:, boundary_point:boundary_point*2] = 126.0
        feature[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = embedded_feature
        return feature

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature = np.zeros((shape[0], shape[1]), dtype=np.float)
        embedded_feature = np.zeros((self.size[0], self.size[1]), dtype=np.float)
        embedded_feature[:, :] = 126.0
        mid_point_y = int(self.size[0] / 2)
        mid_point_x = int(self.size[1] / 2)
        embedded_feature[mid_point_y:self.size[0], 0:mid_point_x] = 255.0
        embedded_feature[0:mid_point_y, mid_point_x:self.size[1]] = 255.0
        feature[self.position[0]:self.position[0] + self.size[0], self.position[1]:self.position[1] + self.size[1]] = embedded_feature
        return feature

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        if self.feat_type == (2, 1):  # two_horizontal
            score = self.evaluate_two_by_one(ii)

        if self.feat_type == (1, 2):  # two_vertical
            score = self.evaluate_one_by_two(ii)

        if self.feat_type == (3, 1):  # three_horizontal
            score = self.evaluate_three_by_one(ii)

        if self.feat_type == (1, 3):  # three_vertical
            score = self.evaluate_one_by_three(ii)

        if self.feat_type == (2, 2):  # four_square
            score = self.evaluate_two_by_two(ii)

        return score

    def evaluate_one_by_three(self, ii):
        boundary_point_x = int(self.size[1] / 3)

        top_left = self.position
        bottom_left = (top_left[0] + self.size[0], top_left[1])

        top_middle_left = (top_left[0], top_left[1] + boundary_point_x)
        bottom_middle_left = (top_left[0] + self.size[0], top_left[1] + boundary_point_x)

        top_middle_right = (top_left[0], top_left[1] + boundary_point_x*2)
        bottom_middle_right = (top_left[0] + self.size[0], top_left[1] + boundary_point_x*2)

        top_right = (top_left[0], top_left[1] + self.size[1])
        bottom_right = (top_left[0] + self.size[0], top_left[1] + self.size[1])

        # sum = D - C - B + A
        left_sum = ii[d(bottom_middle_left)] - ii[d(bottom_left)] - ii[d(top_middle_left)] + ii[d(top_left)]
        middle_sum = ii[d(bottom_middle_right)] - ii[d(bottom_middle_left)] - ii[d(top_middle_right)] + ii[d(top_middle_left)]
        right_sum = ii[d(bottom_right)] - ii[d(bottom_middle_right)] - ii[d(top_right)] + ii[d(top_middle_right)]

        score = left_sum + right_sum - middle_sum
        return score

    def evaluate_three_by_one(self, ii):
        boundary_point_y = int(self.size[0] / 3)

        top_left = self.position
        top_middle_left = (top_left[0] + boundary_point_y, top_left[1])
        bottom_middle_left = (top_left[0] + boundary_point_y*2, top_left[1])
        bottom_left = (top_left[0] + self.size[0], top_left[1])

        top_right = (top_left[0], top_left[1] + self.size[1])
        top_middle_right = (top_left[0] + boundary_point_y, top_left[1] + self.size[1])
        bottom_middle_right = (top_left[0] + boundary_point_y*2, top_left[1] + self.size[1])
        bottom_right = (top_left[0] + self.size[0], top_left[1] + self.size[1])

        # sum = D - C - B + A
        top_sum = ii[d(top_middle_right)] - ii[d(top_middle_left)] - ii[d(top_right)] + ii[d(top_left)]
        middle_sum = ii[d(bottom_middle_right)] - ii[d(bottom_middle_left)] - ii[d(top_middle_right)] + ii[d(top_middle_left)]
        bottom_sum = ii[d(bottom_right)] - ii[d(bottom_left)] - ii[d(bottom_middle_right)] + ii[d(bottom_middle_left)]

        score = top_sum + bottom_sum - middle_sum
        return score

    def evaluate_one_by_two(self, ii):
        mid_point_x = int(self.size[1] / 2)

        top_left = self.position
        bottom_left = (top_left[0] + self.size[0], top_left[1])

        top_middle = (top_left[0], top_left[1] + mid_point_x)
        bottom_middle = (top_left[0] + self.size[0], top_left[1] + mid_point_x)

        top_right = (top_left[0], top_left[1] + self.size[1])
        bottom_right = (top_left[0] + self.size[0], top_left[1] + self.size[1])

        # sum = D - C - B + A
        left_white_pixel_sum = ii[d(bottom_middle)] - ii[d(bottom_left)] - ii[d(top_middle)] + ii[d(top_left)]
        right_gray_pixel_sum = ii[d(bottom_right)] - ii[d(bottom_middle)] - ii[d(top_right)] + ii[d(top_middle)]

        return left_white_pixel_sum - right_gray_pixel_sum

    def evaluate_two_by_one(self, ii):
        mid_point_y = int(self.size[0] / 2)

        top_left = self.position
        middle_left = (top_left[0] + mid_point_y, top_left[1])
        bottom_left = (top_left[0] + self.size[0], top_left[1])

        top_right = (top_left[0], top_left[1] + self.size[1])
        middle_right = (top_left[0] + mid_point_y, top_left[1] + self.size[1])
        bottom_right = (top_left[0] + self.size[0], top_left[1] + self.size[1])

        # sum = D - C - B + A
        top_sum = ii[d(middle_right)] - ii[d(middle_left)] - ii[d(top_right)] + ii[d(top_left)]
        bottom_sum = ii[d(bottom_right)] - ii[d(bottom_left)] - ii[d(middle_right)] + ii[d(middle_left)]

        return top_sum - bottom_sum

    def evaluate_two_by_two(self, ii):
        mid_point_y = int(self.size[0] / 2)
        mid_point_x = int(self.size[1] / 2)

        top_left = self.position
        middle_left = (top_left[0] + mid_point_y, top_left[1])
        bottom_left = (top_left[0] + self.size[0], top_left[1])

        top_middle = (top_left[0], top_left[1] + mid_point_x)
        middle_middle = (top_left[0] + mid_point_y, top_left[1] + mid_point_x)
        bottom_middle = (top_left[0] + self.size[0], top_left[1] + mid_point_x)

        top_right = (top_left[0], top_left[1] + self.size[1])
        middle_right = (top_left[0] + mid_point_y, top_left[1] + self.size[1])
        bottom_right = (top_left[0] + self.size[0], top_left[1] + self.size[1])

        # sum = D - C - B + A
        top_left_sum = ii[d(middle_middle)] - ii[d(middle_left)] - ii[d(top_middle)] + ii[d(top_left)]
        bottom_left_sum = ii[d(bottom_middle)] - ii[d(bottom_left)] - ii[d(middle_middle)] + ii[d(middle_left)]

        top_right_sum = ii[d(middle_right)] - ii[d(middle_middle)] - ii[d(top_right)] + ii[d(top_middle)]
        bottom_right_sum = ii[d(bottom_right)] - ii[d(bottom_middle)] - ii[d(middle_right)] + ii[d(middle_middle)]

        score = bottom_left_sum + top_right_sum - top_left_sum - bottom_right_sum
        return score


def d(shape):
    return shape[0] - 1, shape[1] - 1


def to_integral_image(image):
    integral = np.zeros_like(image, dtype=np.float)

    h = image.shape[0]
    w = image.shape[1]

    for i in range(h):
        for j in range(w):
            top_value = 0.0
            left_value = 0.0
            diagonal_value = 0.0
            current_value = image[i, j]
            if i > 0:
                top_value = integral[i-1, j]
            if j > 0:
                left_value = integral[i, j-1]
            if i > 0 and j > 0:
                diagonal_value = integral[i-1, j-1]

            integral[i, j] = current_value + top_value + left_value - diagonal_value

    return integral


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_images = [to_integral_image(img) for img in images]

    return integral_images


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        np.save("scores.npy", scores, allow_pickle=True)
        # scores = np.load("scores.npy")

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):
            # normalize
            weights = weights / np.sum(weights)

            h_of_x = VJ_Classifier(scores, self.labels, weights)
            h_of_x.train()
            e_t = h_of_x.error
            B_t = e_t / (1 - e_t)
            a_t = np.log(1/B_t)
            self.classifiers.append(h_of_x)
            self.alphas.append(a_t)

            # update weights
            predictions = [h_of_x.predict(x) for x in scores]
            incorrect_indices = np.where(self.labels != predictions)
            b_array = np.zeros_like(self.labels, dtype=np.float)
            b_array[incorrect_indices] = 1.0
            b_array[b_array == 0] = B_t
            temp_weights = np.multiply(weights, b_array)
            weights = temp_weights
            print("Training: ", str(i))

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        feature_index_list = [clf.feature for clf in self.classifiers]

        for i, im in enumerate(ii):
            scores[i, feature_index_list] = [self.haarFeatures[hf_index].evaluate(im) for hf_index in feature_index_list]

        # print("2")

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        alpha_sum = np.sum(self.alphas)
        for x in scores:
            local_sum = 0.0
            for i in range(len(feature_index_list)):
                clf = self.classifiers[i]
                alpha = self.alphas[i]
                local_sum += clf.predict(x) * alpha

            if local_sum >= alpha_sum / 2:
                result.append(1)
            else:
                result.append(-1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h = gray.shape[0]
        w = gray.shape[1]

        x_cumulative = 0.0
        y_cumulative = 0.0
        count = 0.0

        for i in range(12, h - 12, 1):
            for j in range(12, w - 12, 1):
                sub_window = gray[i-12:i+12, j-12:j+12]

                result = self.predict([sub_window])[0]
                if result == 1:
                    x_cumulative += j
                    y_cumulative += i
                    count += 1.0

        x = int(x_cumulative / count)
        y = int(y_cumulative / count)

        image_w_rectangle = cv2.rectangle(image, (x-12, y-12), (x+12, y+12), [0, 0, 255], 2)
        cv2.imwrite(filename + ".png", image_w_rectangle)




