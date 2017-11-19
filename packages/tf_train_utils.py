import sklearn.utils as skutil
import numpy as np
import random as rnd
import tensorflow as tf
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator


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


class ImagePreprocessor(object):
    """
    This class wraps some methods for image preprocessing and normalization.
    """

    def __init__(self, train_images=None):
        """
        If train_images is filled with the array of training images, then internally
        the mean-image and stddev-image are calculated. The function clear_mean_stddev() is going to work ! :)

        :param train_images: An array of n training images with shape like [n, 32, 32, 3]
        """
        self._mean_image = None
        self._stddev_image = None

        if train_images is not None:
            self._mean_image = np.mean(train_images, axis=0)
            self._stddev_image = np.std(train_images, axis=0)

    def clear_mean_stddev(self, x):
        if self._mean_image is None or self._stddev_image is None:
            raise Exception('No mean image or stddev image available.')

        x = x - self._mean_image
        x = x / self._stddev_image
        return x

    @staticmethod
    def normalize_center(x):
        return (x - 128.) / 128.

    @staticmethod
    def apply_clahe(x):
        clahe = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(4, 4))
        r = clahe.apply(x.astype(np.uint8)[:, :, 0])
        g = clahe.apply(x.astype(np.uint8)[:, :, 1])
        b = clahe.apply(x.astype(np.uint8)[:, :, 2])
        return np.dstack([r, g, b]).astype(np.float32)


class TrainSaver(object):
    def __init__(self, directory):
        self._dir = directory
        self._saver = tf.train.Saver()
        self._val_loss = None

    def record(self, session, step, loss):
        if self._val_loss is None or self._val_loss > loss:
            print('Loss decreased from ', self._val_loss, ' to ', loss)
            print('Saving Snapshot to ', self._dir, ' ...')
            print(' ')
            self._val_loss = loss
            self._saver.save(session, os.path.join(self._dir, 'checkpt-'+str(loss)), global_step=step)


class BasicDataAugmenter(object):
    """
    This class wraps the Keras imageDataGenerator for easy image-data augmentation.
    """
    def __init__(self,
                 rotation_range: int=0,
                 width_shift_range: float=0.0,
                 height_shift_range: float=0.0,
                 intensity_shift: float=0.0,
                 shear_range: float=0.0,
                 zoom_range: float=0.0):
        self._intensity_shift = intensity_shift
        self._gen = ImageDataGenerator(rotation_range=rotation_range,
                                       width_shift_range=width_shift_range,
                                       height_shift_range=height_shift_range,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       fill_mode='reflect',
                                       cval=0.0)

    def process(self, x):

        if self._intensity_shift != 0.0:
            if rnd.choice([True, False, True, False]):
                x = x * rnd.uniform(1.0 - self._intensity_shift, 1.0 + self._intensity_shift)
                x = np.clip(x, 0.0, 255.0)
        return self._gen.random_transform(x)


class BatchGenerator(object):
    """ This class implements a simple batch generator. """
    def __init__(self, batch_size, n_classes, x_list, y_list, augmentation_fn=None, preprocessing_fn=None, shuffle=True):
        self._x = np.array(x_list, dtype=np.float32)
        self._y = np.array(y_list, dtype=np.int32)

        assert(len(self._x) == len(self._y))

        self._shuffle = shuffle
        if self._shuffle:
            self._x, self._y = skutil.shuffle(self._x, self._y)

        self._augmentation_fn = augmentation_fn
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

        # Allocate a copy of this batch
        batch_x_cp = np.zeros(shape=batch_x.shape, dtype=np.float32)

        # Push the preprocessed images in this new container
        for i in range(len(batch_x)):
            batch_x_cp[i] = batch_x[i]

            # Do augmentation if function is set
            if self._augmentation_fn is not None:
                batch_x_cp[i] = self._augmentation_fn(batch_x_cp[i])

            # Do preprocessing if function is set
            if self._preprocessing is not None:
                batch_x_cp[i] = self._preprocessing(batch_x_cp[i])

        return batch_x_cp, batch_y
