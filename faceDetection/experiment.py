"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "./"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FACES94_DIR = os.path.join(INPUT_DIR, 'faces94')
POS_DIR = os.path.join(INPUT_DIR, "pos2")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
NEG2_DIR = os.path.join(INPUT_DIR, "neg2")


def generate_examples():
    image = cv2.imread("input_images/man.jpeg")
    image = cv2.resize(image, (120, 60))

    x = 62
    y = 27
    t = 12

    pos = []
    neg = []
    for i in range(5):
        for j in range(5):
            x_ = x - i
            y_ = y - j
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            pos.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

            x_ = x + i
            y_ = y + j
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            pos.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

    h = image.shape[0]
    w = image.shape[1]
    for i in range(15, h - 15, 5):
        for j in range(20, w - 20, 5):
            x_ = j
            y_ = i
            distance = np.sqrt( (x-x_)**2 + (y-y_)**2)
            if distance < 20:
                continue
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            neg.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

    return pos, neg


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs


# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)

        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    two_d = np.reshape(x_mean, size)
    min_value = two_d.min()
    max_value = two_d.max()
    normalized = (two_d - min_value) / (max_value - min_value) * 255
    resized = cv2.resize(normalized, new_dims)
    return resized


def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)

    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")


def evaluate_pca(k, p):
    size = [32, 32]
    X, y = ps6.load_images(YALE_FACES_DIR, size)
    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

    # training
    mu = ps6.get_mean_face(Xtrain)
    eig_vecs, eig_vals = ps6.pca(Xtrain, k)
    Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)

    # testing
    mu = ps6.get_mean_face(Xtest)
    Xtest_proj = np.dot(Xtest - mu, eig_vecs)

    good = 0
    bad = 0

    for i, obs in enumerate(Xtest_proj):

        dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]

        idx = np.argmin(dist)
        y_pred = ytrain[idx]

        if y_pred == ytest[i]:
            good += 1

        else:
            bad += 1

    # print('Good predictions = ', good, 'Bad predictions = ', bad)
    # print('{0:.2f}% accuracy'.format(100 * float(good) / (good + bad)))
    return 100 * float(good) / (good + bad)


def evaluate_pca_params(k, p, iterations):
    cumulative_result = 0.0
    for i in range(iterations):
        cumulative_result += evaluate_pca(k, p)

    return cumulative_result / float(iterations)


def part_1c():
    iterations = 10

    for p in [0.3, 0.5, 0.7, 0.9]:
        for k in [3, 5, 7, 9]:
            print("Evaluating", "p: ", p, "k: ", k)
            result = evaluate_pca_params(k, p, iterations)
            print("Accuracy: ", '{0:.2f}% accuracy'.format(result), "k: ", k, "p: ", p)


def evaluate_faces_94(p, num_iter):
    y0 = 1
    y1 = 2

    X, y = ps6.load_images(FACES94_DIR)

    # Select only the y0 and y1 classes
    idx = y == y0
    idx |= y == y1

    X = X[idx,:]
    y = y[idx]

    # Label them 1 and -1
    y0_ids = y == y0
    y1_ids = y == y1
    y[y0_ids] = 1
    y[y1_ids] = -1

    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytrain)))
    matching_indices = np.where(rand_y == ytrain)[0]
    rand_train_accuracy = matching_indices.shape[0] / ytrain.shape[0]
    # print('(Random) Training accuracy: {0:.2f}%'.format(rand_train_accuracy))

    # Using Weak Classifier
    uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
    wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
    wk_clf.train()
    wk_results = [wk_clf.predict(x) for x in Xtrain]
    matching_indices = np.where(wk_results == ytrain)[0]
    wk__train_accuracy = matching_indices.shape[0] / ytrain.shape[0]
    # print('(Weak) Training accuracy {0:.2f}%'.format(wk__train_accuracy))

    boost = ps6.Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_train_accuracy = 100 * float(good) / (good + bad)
    # print('(Boosting) Training accuracy {0:.2f}%'.format(boost_train_accuracy))

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytest)))
    matching_indices = np.where(rand_y == ytest)[0]
    rand_test_accuracy = matching_indices.shape[0] / ytest.shape[0]
    # print('(Random) Testing accuracy: {0:.2f}%'.format(rand_test_accuracy))

    # Using Weak Classifier
    wk_results = [wk_clf.predict(x) for x in Xtest]
    matching_indices = np.where(wk_results == ytest)[0]
    wk_test_accuracy = matching_indices.shape[0] / ytest.shape[0]
    # print('(Weak) Testing accuracy {0:.2f}%'.format(wk_test_accuracy))

    y_pred = boost.predict(Xtest)
    matching_indices = np.where(y_pred == ytest)[0]
    boost_test_accuracy = matching_indices.shape[0] / ytest.shape[0]
    # print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_test_accuracy))

    return rand_train_accuracy, wk__train_accuracy, boost_train_accuracy, rand_test_accuracy, wk_test_accuracy, boost_test_accuracy


def evaluate_avereage_faces_94(p, num_iter):
    r_train_cumulative = 0.0
    wk_train_cumulative = 0.0
    boost_train_cumulative = 0.0
    r_test_cumulative = 0.0
    wk_test_cumulative = 0.0
    boost_test_cumulative = 0.0

    n = 1

    for i in range(n):
        r_train, wk_train, boost_train, rand_test, wk_test, boost_test = evaluate_faces_94(p, num_iter)
        r_train_cumulative += r_train
        wk_train_cumulative += wk_train
        boost_train_cumulative += boost_train
        r_test_cumulative += rand_test
        wk_test_cumulative += wk_test
        boost_test_cumulative += boost_test

    r_train_f = r_train_cumulative / float(n)
    wk_train_f = wk_train_cumulative / float(n)
    boost_train_f = boost_train_cumulative / float(n)
    rand_test_f = r_test_cumulative / float(n)
    wk_test_f = wk_test_cumulative / float(n)
    boost_test_f = boost_test_cumulative / float(n)

    print("Random Training: ", '{0:.2f}%'.format(r_train_f),
          "Weak Training: ", '{0:.2f}%'.format(wk_train_f),
          "Boost Training: ", '{0:.2f}%'.format(boost_train_f),
          "Random Test: ", '{0:.2f}%'.format(rand_test_f),
          "Weak Test: ", '{0:.2f}%'.format(wk_test_f),
          "Boost Test: ", '{0:.2f}%'.format(boost_test_f))


def part_2a():

    for p in [0.3, 0.5, 0.7, 0.9]:
        for k in [2, 4, 8, 12]:
            evaluate_avereage_faces_94(p, k)


def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""

    feature1 = ps6.HaarFeature((2, 1), (25, 30), (50, 100))
    feature1.preview((200, 200), filename="ps6-3-a-1.png")

    feature1 = ps6.HaarFeature((1, 2), (10, 25), (50, 150))
    feature1.preview((200, 200), filename="ps6-3-a-2.png")

    feature1 = ps6.HaarFeature((3, 1), (50, 50), (100, 150))
    feature1.preview((200, 200), filename="ps6-3-a-3.png")

    feature1 = ps6.HaarFeature((1, 3), (50, 125), (100, 50))
    feature1.preview((200, 200), filename="ps6-3-a-4.png")

    feature1 = ps6.HaarFeature((2, 2), (50, 25), (100, 150))
    feature1.preview((200, 200), filename="ps6-3-a-5.png")


def part_4_a_b():

    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)

    train_pos = pos[:35]
    train_neg = neg[:]
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(5)

    VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")

    predictions = VJ.predict(images)
    matching_indices = np.where(predictions == labels)[0]
    vj_accuracy = matching_indices.shape[0] / labels.shape[0]
    print("Prediction accuracy on training: {0:.2f}%".format(vj_accuracy))

    neg = load_images_from_dir(NEG2_DIR)

    test_pos = pos[35:]
    test_neg = neg[:35]
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)

    matching_indices = np.where(predictions == real_labels)[0]
    vj_accuracy = matching_indices.shape[0] / real_labels.shape[0]
    print("Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy))


def  part_4_c():

    pos, neg = generate_examples()

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(pos, neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(1)

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    VJ.faceDetection(image, filename="ps6-4-c-1")


if __name__ == "__main__":
    # part_1a_1b()
    # part_1c()
    part_2a()
    # part_3a()
    # part_4_a_b()
    # part_4_c()
