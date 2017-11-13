import tensorflow as tf


class TfModel(object):
    def __init__(self, input_shape,
                 training_phase: int = 1,
                 dropout_keep_prob = 0.72,
                 kernel_initializer=tf.truncated_normal_initializer(),
                 kernel_regularization=tf.contrib.layers.l2_regularizer()):

        self._input_shape = input_shape
        self._training_phase = training_phase
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularization
        self._dropout_keep_prob = dropout_keep_prob

    def __conv2d(self, x, name, filters, kernel_size, padding='same', strides=(1, 1)):
        w = tf.get_variable(name+'_weights',
                            shape=[kernel_size[0],
                                   kernel_size[1],
                                   x.shape[3],
                                   filters],
                            initializer=self._kernel_initializer,
                            regularizer=self._kernel_regularizer,
                            dtype=tf.float32)

        x = tf.nn.conv2d(x, w, strides=[1, strides[0], strides[1], 1], padding=padding)
        x = tf.nn.bias_add(x, tf.Variable(tf.zeros(filters)))
        return x

    def __fc_layer(self, x, name, neurons):
        w = tf.get_variable(name+'_weights',
                            shape=[int(x.shape[0]), neurons],
                            initializer=self._kernel_initializer,
                            dtype=tf.float32)

        x = tf.add(tf.matmul(x, w), tf.Variable(tf.zeros(neurons)))
        return x

    def __relu(self, x):
        return tf.nn.relu(x)

    def __elu(self, x):
        return tf.nn.elu(x)

    def __leaky_relu(self, x):
        return tf.nn.leaky_relu(x)

    def __max_pooling2d(self, x, pool_size=(2, 2), strides=(2, 2)):
        return tf.nn.max_pool(x,
                              ksize=[1, pool_size[0], pool_size[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding='SAME')

    def __dropout(self, x, keep_prob):
        return tf.nn.dropout(x, tf.constant(keep_prob, dtype=tf.float32))

    def __batch_norm(self, x):
        return tf.contrib.layers.batch_normalization(x, training=bool(self._training_phase))

    def __flatten(self, x):
        return tf.contrib.layers.flatten(x)


    def construct(self, input_tensor):
        raise NotImplemented('Base class tfModel wont implement construct')


class TfLeNet(TfModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def construct(self, input_tensor):

        # Input = 32x32xC
        x = self.__conv2d(input_tensor, 'conv1', filters=6, kernel_size=(5, 5), padding='valid')
        x = self.__relu(x)

        # Layer 1: Input = 28x28x6. Output = 14x14x6.
        x = self.__max_pooling2d(x)

        # Layer 2: Convolutional. Output = 10x10x16.
        x = self.__conv2d(x, 'conv2', filters=16, kernel_size=(5, 5), padding='valid')
        x = self.__relu(x)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = self.__max_pooling2d(x)

        # Flatten. Input = 5x5x16. Output = 400.
        x = self.__flatten(x)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        x = self.__fc_layer(x, 'fc1', 120)

        # Activation.
        x = self.__relu(x)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        x = self.__fc_layer(x, 'fc2', 84)

        # Activation + Dropout.
        x = self.__relu(x)
        x = self.__dropout(x, self._dropout_keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        logits = self.__fc_layer(x, 'logits', 10)
        return logits