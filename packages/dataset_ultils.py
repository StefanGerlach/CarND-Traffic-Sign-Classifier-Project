import pickle
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import cv2


def load_datasets(training_file, validation_file, testing_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_translation_file(translation_file):
    with open(translation_file, mode='r') as f:
        csv_file = pandas.read_csv(f)

    return dict([(x[1][0], x[1][1]) for x in csv_file.iterrows()])


def print_datasets_stats(x_train, x_valid, x_test, translation_file):
    n_train = len(x_train)
    n_valid = len(x_valid)
    n_test = len(x_test)
    n_classes = len(translation_file)

    print("Number of Training examples = ", n_train)
    print("Number of Validation examples = ", n_valid)
    print("Number of Test examples = ", n_test)
    print("Unique classes = ", n_classes)

    return n_train, n_valid, n_test, n_classes


def visualize_predictions(predictions: dict, title: str):
    """
    This function will plot the predictions in the predictions-dict
    """
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')

    n_cols = 10
    n_rows = 3

    c_img = 0
    c_sub = 1
    for k in predictions:

        fig.add_subplot(n_rows, n_cols, c_sub)
        plt.axis('off')

        img = cv2.resize((predictions[k][0] + 1.0) / 2.0, dsize=(256, 256))
        plt.imshow(img)
        c_img += 1
        c_sub += 1
        if c_img % (n_cols * n_rows) == 0:
            plt.show()
            fig.show()

            c_sub = 1
            fig = plt.figure()
            plt.title(title)
            plt.axis('off')


def visualize_dataset_content(x, y, y_translation, n_samples=5):
    """
    This function will plot n_samples in a random order of every class.
    :param n_samples: how many samples to plot per class
    :param y_translation: the class strings
    :param x: the images in an array
    :param y: the corresponding labels
    """

    indices_mapping = {}
    # Sorting into class-based map/dictionary
    for idx, y_hat in enumerate(y):
        if y_hat not in indices_mapping:
            indices_mapping[y_hat] = []
        indices_mapping[y_hat].append(idx)

    # Random Sampling
    for y_hat in indices_mapping:
        how_many = n_samples
        if len(indices_mapping[y_hat]) < n_samples:
            how_many = len(indices_mapping[y_hat])
        indices_mapping[y_hat] = rnd.sample(indices_mapping[y_hat], how_many)

    for y_hat in indices_mapping:
        # Visualize in a plot
        fig = plt.figure()

        plot_rows = 1
        plot_cols = n_samples

        current_image = 1

        for idx in indices_mapping[y_hat]:
            # For every image, create a subplot
            fig.suptitle(y_translation[y_hat])
            fig.add_subplot(plot_rows, plot_cols, current_image)

            img = cv2.resize(x[idx], dsize=(128, 128))
            plt.imshow(img)
            plt.axis('off')
            current_image += 1

        plt.show()
    return


def visualize_dataset_frequencies(y, y_translation):
    # count the frequencies of classes in dataset and visualize
    hist = {}

    for label_id in y:
        if label_id not in y_translation:
            raise Exception('label_id not found in translation file.')

        l = y_translation[label_id]
        if l not in hist:
            hist[l] = 0
        hist[l] += 1

    # visualize as histogram
    fig = plt.figure(figsize=(16, 12))
    sub = fig.add_subplot(1, 1, 1)
    sub.set_title('Histogram of classes')
    y_data = np.array([float(hist[k]) for k in hist])
    plt.bar(range(len(hist)), y_data, align='center')
    x_axis = np.array([k for k in hist])
    plt.xticks(range(len(hist)), x_axis, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.show()


def visualize_wrong_classifications(x, y_gt, y_pred, y_translations):
    pass
