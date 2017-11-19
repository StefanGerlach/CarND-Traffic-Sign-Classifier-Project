import pickle
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import glob
import cv2
import os


def create_dataset(src_path: str, dst_filename: str):
    """
    This function will iterate over a directory to create a pickle dataset-file
    The label is read from the filename with a specific convention!

    filename = my_file_name_<label>.extension

    The label is read between the last '_' and the '.' of the file-extension.
    """

    filenames = glob.glob(os.path.join(src_path, '*'))
    if len(filenames) <= 0:
        raise Exception('No files found in directory ' + str(src_path))

    dataset = {'features': [],
               'labels': []}

    for filename in filenames:
        try:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label = int(filename.split('_')[-1].split('.')[0])

            dataset['features'].append(img)
            dataset['labels'].append(label)
        except:
            print('Could not load file ', filename)

    with open(dst_filename, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(file):
    with open(file, mode='rb') as f:
        pickle_file = pickle.load(f)

    x, y = pickle_file['features'], pickle_file['labels']
    return [x, y]


def load_translation_file(translation_file):
    with open(translation_file, mode='r') as f:
        csv_file = pandas.read_csv(f)

    return dict([(x[1][0], x[1][1]) for x in csv_file.iterrows()])


def print_datasets_stats(x, y):
    n_examples = len(x)
    n_classes = len(np.unique(y))

    print("Number of examples = ", n_examples)
    print("Unique classes = ", n_classes)

    return n_examples, n_classes


def visualize_single_prediction(img, title: str, predictions:dict):
    """
    This function is going to plot a single prediction in detail.
    It expects a 3D Tensor for image data, a string-title and a
    dictionary with keys as label-names with probabilities as values.

    E.g. predictions = {'no passing': 0.5277,
                        'stop': 0.012, ...}
    """

    figure = plt.figure()
    figure.tight_layout()
    plt.axis('off')

    sub_plot = figure.add_subplot(121)
    sub_plot.set_title(title)

    buffer = np.zeros_like(img)
    img = cv2.normalize(img, dst=buffer, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    plt.imshow(img)

    sub_plot = figure.add_subplot(122)

    sub_plot.set_title('Prediction probabilities')
    y_data = np.array([float(predictions[label]) for label in predictions])
    plt.bar(range(len(predictions)), y_data, align='center')
    x_axis = np.array([label for label in predictions])
    plt.xticks(range(len(predictions)), x_axis, fontsize=8, rotation='vertical')
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    plt.subplots_adjust(bottom=0.1)


def visualize_predictions(predictions: dict, title: str):
    """
    This function will plot the predictions in the predictions-dict
    """
    fig = plt.figure()
    fig.tight_layout()

    plt.title(title)
    plt.axis('off')

    n_cols = 10
    n_rows = 3

    c_img = 0
    c_sub = 1
    for k in predictions:

        fig.add_subplot(n_rows, n_cols, c_sub)
        plt.axis('off')

        img = cv2.resize(predictions[k][0], dsize=(256, 256))
        buffer = np.zeros_like(img)
        img = cv2.normalize(img, dst=buffer, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        plt.imshow(img)
        c_img += 1
        c_sub += 1
        if c_img % (n_cols * n_rows) == 0:
            plt.show()
            c_sub = 1
            fig = plt.figure()
            plt.title(title)
            plt.axis('off')

    plt.show()


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
        fig.tight_layout()

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
            plt.subplots_adjust(top=0.95)
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
    fig.tight_layout()
    sub = fig.add_subplot(1, 1, 1)
    sub.set_title('Histogram of classes')
    y_data = np.array([float(hist[k]) for k in hist])
    plt.bar(range(len(hist)), y_data, align='center')
    x_axis = np.array([k for k in hist])
    plt.xticks(range(len(hist)), x_axis, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.show()

