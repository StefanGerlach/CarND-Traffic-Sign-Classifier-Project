import pickle
import pandas
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_dataset(x, y, y_translation):
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
