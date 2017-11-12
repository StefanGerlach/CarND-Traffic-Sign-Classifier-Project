import tensorflow as tf


class TfModel(object):
    def __init__(self, input_shape, training_phase: int = 1):
        self._input_shape = input_shape
        self._training_phase = training_phase

    def __conv2(self, x, filters, kernel_size, strides, dilation_rate):
        tf.layers.conv2d(x, filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         dilation_rate=dilation_rate)

    def __batchnorm(self, x):
        return tf.layers.batch_normalization(x, training=bool(self._training_phase))

    def construct(self, input_tensor):
        raise NotImplemented('Base class tfModel wont implement construct')


class TfLeNet(TfModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def construct(self, input_tensor):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1

        x = tf.nn.conv2d(input_tensor,
                         tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma)),
                         strides=[1, 1, 1, 1],
                         padding='VALID')

        x = tf.nn.bias_add(x, tf.Variable(tf.zeros(6)))
        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        x = tf.nn.conv2d(x, tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma)),
                         strides=[1, 1, 1, 1],
                         padding='VALID')

        x = tf.nn.bias_add(x, tf.Variable(tf.zeros(16)))
        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        x = tf.contrib.layers.flatten(x)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        x = tf.add(tf.matmul(x, tf.Variable(tf.truncated_normal(shape=[400, 120], mean=mu, stddev=sigma))),
                   tf.Variable(tf.zeros(120)))

        # TODO: Activation.
        x = tf.nn.relu(x)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        x = tf.add(tf.matmul(x, tf.Variable(tf.truncated_normal(shape=[120, 84], mean=mu, stddev=sigma))),
                   tf.Variable(tf.zeros(84)))

        # TODO: Activation.
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, tf.constant(0.75, dtype=tf.float32))

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        logits = tf.add(tf.matmul(x, tf.Variable(tf.truncated_normal(shape=[84, 10], mean=mu, stddev=sigma))),
                        tf.Variable(tf.zeros(10)))

        return logits