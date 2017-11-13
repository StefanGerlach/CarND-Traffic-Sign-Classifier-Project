import tensorflow as tf
import tqdm
import packages.dataset_ultils as util
import packages.tf_models as models
import packages.tf_train_utils as train_utils


def preprocess_image(x):
    return (x - 128.) / 128.


def cross_entropy_loss(y_pred, y):
    cross_ent = -tf.reduce_sum(y * tf.log(y_pred), axis=1)
    loss = tf.reduce_mean(cross_ent)
    return loss


def accuracy(y_pred, y):
    equal = tf.equal(tf.arg_max(y_pred, -1), tf.argmax(y, -1))
    return tf.reduce_mean(tf.cast(equal, tf.float32))


def optimizer():
    return tf.train.AdamOptimizer(learning_rate=base_lr)


def initialize_tf_session():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return session


train_x, train_y, valid_x, valid_y, test_x, test_y = util.load_datasets('datasets/train.p', 'datasets/valid.p', 'datasets/test.p')
translations = util.load_translation_file('signnames.csv')

# util.visualize_dataset(train_y, translations)
# util.visualize_dataset(valid_y, translations)
# util.visualize_dataset(test_y, translations)
_, _, _, num_class = util.print_datasets_stats(train_x, valid_x, test_x, translations)

# Make sure that every class occurence-probability is nearly equal
class_equalizer = train_utils.ClassEqualizer(train_x, train_y)
# Fill images up with random samples
eq_train_x, eq_train_y = class_equalizer.fill_up_with_copies()

n_classes = num_class
base_lr = 1e-3
batch_size = 256
epochs = 50
steps_per_epoch = int(len(eq_train_y) / batch_size)

# Build the BatchGenerator
batch_generator = train_utils.BatchGenerator(batch_size=batch_size,
                                             n_classes=n_classes,
                                             x_list=eq_train_x,
                                             y_list=eq_train_y,
                                             preprocessing_fn=preprocess_image)

# x is the placeholder for a batch of images
x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)

# y is the placeholder for a batch of labels
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)

# model is our deep neural network architecture
model_predictions = models.TfLeNet(input_shape=[None, 32, 32, 3],
                                   n_classes=n_classes,
                                   kernel_regularization=0.0).construct(x)

# define the loss function
loss = cross_entropy_loss(model_predictions, y)

# define the acc function
acc = accuracy(model_predictions, y)

# define the optimizer
target_op = optimizer().minimize(loss)

loss_history = []

# initialize all variables and go
sess = initialize_tf_session()

for epoch in range(epochs):

    # iterate n steps until one complete epoch was shown to the network
    for i in tqdm.tqdm(range(steps_per_epoch)):

        # get the batch from batch generator
        batch_x, batch_y = batch_generator.next()

        # run a optimizer step
        _, current_loss, current_acc = sess.run([target_op, loss, acc], feed_dict={x: batch_x, y: batch_y})
        print(current_loss, current_acc)





