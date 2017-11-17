import tensorflow as tf
import numpy as np
import os
import packages.dataset_ultils as util
import packages.tf_models as models
import packages.tf_train_utils as train_utils


def preprocess_image(x):
    return (x - 128.) / 128.


def cross_entropy_loss(y_pred, y):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(cross_entropy)


def accuracy(y_pred, y):
    equal = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(equal, tf.float32))


def optimizer():
    return tf.train.AdamOptimizer(learning_rate=base_lr)


train_x, train_y, valid_x, valid_y, test_x, test_y = util.load_datasets('datasets/train.p', 'datasets/valid.p', 'datasets/test.p')
translations = util.load_translation_file('signnames.csv')

# util.visualize_dataset(train_y, translations)
# util.visualize_dataset(valid_y, translations)
# util.visualize_dataset(test_y, translations)
_, _, _, num_class = util.print_datasets_stats(train_x, valid_x, test_x, translations)

# Make sure that every class occurrence-probability is nearly equal
class_equalizer = train_utils.ClassEqualizer(train_x, train_y)

# Fill images up with random samples
eq_train_x, eq_train_y = class_equalizer.fill_up_with_copies()

logdir = 'logs'
experiment_name = 'squeeze_softmax_xavier_drop_v12_fc'
base_lr = 1e-3
batch_size = 128
epochs = 50


n_classes = num_class
steps_per_epoch = int(np.ceil(len(eq_train_y) / float(batch_size)))
steps_per_epoch_valid = int(np.ceil(len(valid_y) / float(batch_size)))

# Build the BatchGenerator for training
train_batch_generator = train_utils.BatchGenerator(batch_size=batch_size,
                                                   n_classes=n_classes,
                                                   x_list=eq_train_x,
                                                   y_list=eq_train_y,
                                                   preprocessing_fn=preprocess_image,
                                                   shuffle=True)

# Build the BatchGenerator for validation
valid_batch_generator = train_utils.BatchGenerator(batch_size=batch_size,
                                                   n_classes=n_classes,
                                                   x_list=valid_x,
                                                   y_list=valid_y,
                                                   preprocessing_fn=preprocess_image,
                                                   shuffle=False)

# x is the placeholder for a batch of images
x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)

# y is the placeholder for a batch of labels
y = tf.placeholder(shape=[None], dtype=tf.int32)
y_one_hot = tf.one_hot(y, num_class)

# Model is our deep neural network architecture
# model_predictions = models.TfLeNet(input_shape=[None, 32, 32, 3],
#                                    n_classes=n_classes,
#                                    kernel_regularization=0.0,
#                                    dropout_keep_prob=0.75).construct(x)

model_predictions = models.TfCustomSqueezeNet(input_shape=[None, 32, 32, 3],
                                              n_classes=n_classes,
                                              kernel_regularization=1e-3,
                                              dropout_keep_prob=0.6).construct(x)

# Define the loss function
loss = cross_entropy_loss(model_predictions, y_one_hot)

# Define the acc function
acc = accuracy(model_predictions, y_one_hot)

# Define the optimizer
target_op = optimizer().minimize(loss)

# Creating some history container
train_loss_history = []
train_acc_history = []

validation_loss_history = []
validation_acc_history = []

with tf.Session() as sess:

    # Create summaries for this run
    summaries = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    summary_placeholder = []
    summary_ops = []

    for s in summaries:
        summary_placeholder.append(tf.placeholder(dtype=tf.float32))
        summary_ops.append(tf.summary.scalar(s, summary_placeholder[-1]))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Initialize a Tensorboard summary Writer
    summary_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, experiment_name),
                                           graph=sess.graph)

    # Run over epochs
    for epoch in range(epochs):
        print('------------------------------')
        print('epoch ', epoch)
        print('------------------------------')

        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0

        # Iterate n steps until one complete epoch was shown to the network
        for i in range(steps_per_epoch):
            # Get the batch from training batch generator
            batch_x, batch_y = train_batch_generator.next()

            # Run a optimizer step
            _, current_train_loss, current_train_acc = sess.run([target_op, loss, acc], feed_dict={x: batch_x, y: batch_y})

            train_loss += current_train_loss
            train_acc += current_train_acc

        # Iterate n steps for the validation now
        for j in range(steps_per_epoch_valid):
            # Get the batch from training batch generator
            batch_x, batch_y = valid_batch_generator.next()

            # Just get the loss and accuracy over this batch
            current_val_loss, current_acc = sess.run([loss, acc], feed_dict={x: batch_x, y: batch_y})

            valid_loss += current_val_loss
            valid_acc += current_acc

        # Remember the history of this run
        train_loss_history.append(train_loss / steps_per_epoch)
        train_acc_history.append(train_acc / steps_per_epoch)

        validation_loss_history.append(valid_loss / steps_per_epoch_valid)
        validation_acc_history.append(valid_acc / steps_per_epoch_valid)

        # I like Tensorboard !
        feed_dict = { summary_placeholder[0]: train_loss_history[-1],
                      summary_placeholder[1]: train_acc_history[-1],
                      summary_placeholder[2]: validation_loss_history[-1],
                      summary_placeholder[3]: validation_acc_history[-1]}

        tf_summaries = sess.run(tf.summary.merge(summary_ops), feed_dict=feed_dict)
        summary_writer.add_summary(tf_summaries, epoch)

        # Print to Console, too
        print('train_loss: ', train_loss_history[-1],
              ' train_acc: ', train_acc_history[-1])

        print('valid_loss: ', validation_loss_history[-1],
              ' valid_acc: ', validation_acc_history[-1])


