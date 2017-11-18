import sklearn.utils as skutil
import numpy as np
import random as rnd


class ClassEqualizer(object):
    """ This class implements a basic class-frequency equalizer. """
    def __init__(self, x_list, y_list):
        self._x = x_list
        self._y = y_list

    def fill_up_with_copies(self):
        # First of all, create a list of item indices with classes
        class_sample_indices = {}
        for i in range(len(self._y)):
            if self._y[i] not in class_sample_indices:
                class_sample_indices[self._y[i]] = []
            # remember the index of the sample with this class
            class_sample_indices[self._y[i]].append(i)

        # Calc the maximum count of a class
        class_max_count = np.max(np.array([len(class_sample_indices[k]) for k in class_sample_indices]))

        # Now, fill up a new dictionary with indices
        class_sample_indices_ext = class_sample_indices

        for k in class_sample_indices_ext:
            needed_for_fill = class_max_count - len(class_sample_indices[k])
            for i in range(needed_for_fill):
                class_sample_indices_ext[k].append(rnd.choice(class_sample_indices[k]))

        # Finally, create a new list in memory with multiple references to original set
        x_list = []
        y_list = []

        for k in class_sample_indices_ext:
            for i in range(len(class_sample_indices_ext[k])):
                x_list.append(self._x[class_sample_indices_ext[k][i]])
                y_list.append(k)

        return x_list, y_list


class BatchGenerator(object):
    """ This class implements a simple batch generator. """
    def __init__(self, batch_size, n_classes, x_list, y_list, preprocessing_fn=None, shuffle=True):
        self._x = np.array(x_list, dtype=np.float32)
        self._y = np.array(y_list, dtype=np.int32)

        assert(len(self._x) == len(self._y))

        self._shuffle = shuffle
        if self._shuffle:
            self._x, self._y = skutil.shuffle(self._x, self._y)

        self._preprocessing = preprocessing_fn
        self._batch_size = batch_size
        self._num_classes = n_classes
        self._index = 0

    def reset(self):
        self._index = 0

    def next(self):
        current_sta_index = self._index
        current_end_index = self._index + self._batch_size

        if current_end_index >= len(self._x):
            current_end_index = len(self._x) - 1
            self._index = 0
            if self._shuffle:
                self._x, self._y = skutil.shuffle(self._x, self._y)
        else:
            self._index += self._batch_size

        batch_x = self._x[current_sta_index:current_end_index]
        batch_y = self._y[current_sta_index:current_end_index]

        # Do preprocessing if function is set
        if self._preprocessing is not None:

            # Allocate a copy of this batch
            batch_x_cp = np.zeros(shape=batch_x.shape, dtype=np.float32)

            # Push the preprocessed images in this new container
            for i in range(len(batch_x)):
                batch_x_cp[i] = self._preprocessing(batch_x[i])

            # Reassign
            batch_x = batch_x_cp

        return batch_x, batch_y
