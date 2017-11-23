import tensorflow as tf
import numpy as np
import os

import sklearn.utils as skutil
import packages.dataset_utils as util
import packages.tf_models as models
import packages.tf_train_utils as train_utils

# For creation of the custom traffic sign images dataset
# util.create_dataset('images/custom_signs', 'custom_test.p')


class ModelTrainer(object):
    def __init__(self, log_directory: str, path_to_traindata: str, path_to_valdata: str, translations_file: str):

        self._logdir = log_directory
        self._path_traindata = path_to_traindata
        self._path_valdata = path_to_valdata
        self._translations = translations_file

        self._training_set = None
        self._validation_set = None

        self._n_classes = None
        self._class_translations = None

        self.class_equalizer = None
        self.image_preprocessor = None
        self.image_augmenter = None

        # the later applied preprocessing function:
        self._preprocessing_function = None

        # the later applied augmentation function
        self._augmentation_function = None

        # the generators
        self._train_batch_generator = None
        self._valid_batch_generator = None

        # the training hyper parameters
        self._learning_rate = None
        self._momentum = None
        self._batch_size = None
        self._steps_per_epoch_train = None
        self._steps_per_epoch_valid = None
        self._epochs = None
        self._optimizer = None

        # the TensorFlow session
        self._session = None
        self._model = None
        self._train_saver = None

        # the output of our model
        self._model_predictions = None
        self._tf_graph_elements = None

        # init sequence
        self.__init_datasets()
        self.__init_dataset_equalizer()
        self.__init_image_preprocessor()
        self.__init_image_augmenter()

    def __init_datasets(self):
        # Initialize the training and validation datasets
        self._training_set = util.load_dataset(self._path_traindata)
        self._validation_set = util.load_dataset(self._path_valdata)

        # Read in the translations for the labels
        self._class_translations = util.load_translation_file(self._translations)

        # Set n_classes
        self._n_classes = len(np.unique(self._training_set[1]))

    def __init_dataset_equalizer(self):
        # Initialize the class, responsible for class frequencies
        self.class_equalizer = train_utils.ClassEqualizer(self._training_set[0],
                                                          self._training_set[1])

    def __init_image_preprocessor(self):
        assert self._training_set
        # Initialize the ImageProcessor with the raw image data
        self.image_preprocessor = train_utils.ImagePreprocessor(self._training_set[0])

    def __init_image_augmenter(self):
        # The arguments for the basic data augmenter may also be arguments for
        # the ModelTrainer for iterative hyper parameter search !
        self.image_augmenter = train_utils.BasicDataAugmenter(rotation_range=20,
                                                              width_shift_range=0.1,
                                                              height_shift_range=0.1,
                                                              intensity_shift=0.75,
                                                              shear_range=0.2,
                                                              zoom_range=0.2)

    def __init_batch_generators(self):
        # This function will initialize the batch generators for training and validation
        assert self._batch_size
        assert self._n_classes
        assert self._training_set
        assert self._validation_set

        self._train_batch_generator = train_utils.BatchGenerator(batch_size=self._batch_size,
                                                                 n_classes=self._n_classes,
                                                                 x_list=self._training_set[0],
                                                                 y_list=self._training_set[1],
                                                                 augmentation_fn=self._augmentation_function,
                                                                 preprocessing_fn=self._preprocessing_function,
                                                                 shuffle=True)

        self._valid_batch_generator = train_utils.BatchGenerator(batch_size=self._batch_size,
                                                                 n_classes=self._n_classes,
                                                                 x_list=self._validation_set[0],
                                                                 y_list=self._validation_set[1],
                                                                 preprocessing_fn=self._preprocessing_function,
                                                                 shuffle=False)

    def get_image_shape(self):
        assert self._training_set
        assert len(self._training_set[0]) > 0
        return self._training_set[0][-1].shape

    def get_num_classes(self):
        assert self._n_classes
        return self._n_classes

    def equalize_traindata_class_frequencies(self):
        # Make sure that every class occurrence-probability is nearly equal
        assert self.class_equalizer
        assert self._training_set

        # Fill images up with random samples
        self._training_set[0], self._training_set[1] = self.class_equalizer.fill_up_with_copies()

    def set_preprocessing_function(self, preprocessing_function):
        self._preprocessing_function = preprocessing_function

    def set_augmentation_function(self, augmentation_function):
        self._augmentation_function = augmentation_function

    def set_training_parameter(self,
                               learning_rate: float=1e-3,
                               momentum: float=0.9,
                               batch_size: int=128,
                               epochs: int=50,
                               optimizer: str='adam'):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._batch_size = batch_size
        self._epochs = epochs
        self._steps_per_epoch_train = int(np.ceil(len(self._training_set[0]) / float(batch_size)))
        self._steps_per_epoch_valid = int(np.ceil(len(self._validation_set[0]) / float(batch_size)))

        if optimizer not in ['adam', 'sgd']:
            raise Exception('Unrecognized optimizer, try adam or sgd.')

        if optimizer == 'adam':
            self._optimizer = train_utils.adam_optimizer(self._learning_rate)

        if optimizer == 'sgd':
            self._optimizer = train_utils.sgd_optimizer(self._learning_rate, self._momentum)

        # At this point we are ready for initialization of the batch generators
        self.__init_batch_generators()

    def set_model(self, model: models.TfModel):
        # Get the traindata image shape
        input_shape = self.get_image_shape()

        # Remember the model
        self._model = model

        # Build a dictionary with TensorFlow graph - nodes
        self._tf_graph_elements = {}

        # x is the placeholder for a batch of images
        x = tf.placeholder(shape=[None,
                                  input_shape[0],
                                  input_shape[1],
                                  input_shape[2]],
                           dtype=tf.float32)

        # y is the placeholder for a batch of labels
        y = tf.placeholder(shape=[None], dtype=tf.int32)

        # The x placeholder
        self._tf_graph_elements.update({'x': x})

        # The y placeholder
        self._tf_graph_elements.update({'y': y})

        # The model dropout placeholder
        self._tf_graph_elements.update({'dropout': model.dropout_keep_prob_placeholder})

        # The One-Hot encoded y
        self._tf_graph_elements.update({'y_one_hot': tf.one_hot(y, self._n_classes)})

        # Set model_predictions to the output of the model
        self._tf_graph_elements.update({'model_predictions': model.construct(x)})

        # The proper probabilities are the logits with softmax-function applied
        self._model_predictions = tf.nn.softmax(self._tf_graph_elements['model_predictions'])

        # Define the loss function op
        self._tf_graph_elements.update({'loss': train_utils.cross_entropy_loss(
            self._tf_graph_elements['model_predictions'],
            self._tf_graph_elements['y_one_hot'])})

        # Define the acc function op
        self._tf_graph_elements.update({'acc': train_utils.accuracy(
            self._tf_graph_elements['model_predictions'],
            self._tf_graph_elements['y_one_hot'])})

        # Define the evaluation function op
        self._tf_graph_elements.update({'evaluate': train_utils.evaluation(
            self._tf_graph_elements['model_predictions'],
            self._tf_graph_elements['y_one_hot'])})

        # Define the optimizer op
        self._tf_graph_elements.update({'target_op': self._optimizer.minimize(self._tf_graph_elements['loss'])})

        # Instantiate the TF Session
        self._session = tf.Session()

        # Initialize all variables
        self._session.run(tf.global_variables_initializer())

    def load_weights(self, path: str):
        saver = tf.train.Saver()
        saver.restore(self._session, path)

    def get_last_checkpoint_filename(self):
        return self._train_saver.last_checkpoint

    def fit(self, experiment_name: str):
        """
        In this function, all components will come together for training the model.
        :param experiment_name: Is the current experiment name
        :param display_most_unsure: After training the n most unsure correct samples are shown.
        """
        assert self._session

        log_dir = os.path.join(self._logdir, experiment_name)

        # Creating some history container
        train_loss_history = []
        train_acc_history = []

        validation_loss_history = []
        validation_acc_history = []

        # Build the TrainSaver
        self._train_saver = train_utils.TrainSaver(directory=log_dir)

        # Create summaries for this run
        summaries = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        summary_placeholder = []
        summary_ops = []

        for s in summaries:
            summary_placeholder.append(tf.placeholder(dtype=tf.float32))
            summary_ops.append(tf.summary.scalar(s, summary_placeholder[-1]))

        # Initialize a Tensorboard summary Writer
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=self._session.graph)

        # Run over epochs
        for epoch in range(self._epochs):
            print('------------------------------')
            print('epoch ', epoch)
            print('------------------------------')

            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0

            # Iterate n steps until one complete epoch was shown to the network
            for i in range(self._steps_per_epoch_train):
                # Get the batch from training batch generator
                batch_x, batch_y = self._train_batch_generator.next()

                # Prepare the graph nodes to run and the graph inputs
                ops_to_run = [self._tf_graph_elements['target_op'],
                              self._tf_graph_elements['loss'],
                              self._tf_graph_elements['acc']]

                feed_dict = {self._tf_graph_elements['x']: batch_x,
                             self._tf_graph_elements['y']: batch_y,
                             self._tf_graph_elements['dropout']: self._model.dropout_training_value}

                # Run a optimizer step
                _, current_train_loss, current_train_acc = self._session.run(ops_to_run, feed_dict=feed_dict)

                train_loss += current_train_loss
                train_acc += current_train_acc

            # Iterate n steps for the validation now
            for j in range(self._steps_per_epoch_valid):
                # Get the batch from training batch generator
                batch_x, batch_y = self._valid_batch_generator.next()

                # Prepare the graph nodes to run and the graph inputs
                ops_to_run = [self._tf_graph_elements['loss'],
                              self._tf_graph_elements['acc']]

                feed_dict = {self._tf_graph_elements['x']: batch_x,
                             self._tf_graph_elements['y']: batch_y,
                             self._tf_graph_elements['dropout']: 1.0}

                # Just get the loss and accuracy over this batch
                current_val_loss, current_acc = self._session.run(ops_to_run, feed_dict=feed_dict)

                valid_loss += current_val_loss
                valid_acc += current_acc

            # Remember the history of this run
            train_loss_history.append(train_loss / self._steps_per_epoch_train)
            train_acc_history.append(train_acc / self._steps_per_epoch_train)

            validation_loss_history.append(valid_loss / self._steps_per_epoch_valid)
            validation_acc_history.append(valid_acc / self._steps_per_epoch_valid)

            # Record this snapshot
            self._train_saver.record(session=self._session, step=epoch, loss=validation_loss_history[-1])

            # I like Tensorboard !
            feed_dict = {summary_placeholder[0]: train_loss_history[-1],
                         summary_placeholder[1]: train_acc_history[-1],
                         summary_placeholder[2]: validation_loss_history[-1],
                         summary_placeholder[3]: validation_acc_history[-1]}

            tf_summaries = self._session.run(tf.summary.merge(summary_ops), feed_dict=feed_dict)
            summary_writer.add_summary(tf_summaries, epoch)

            # Print to Console, too
            print('train_loss: ', train_loss_history[-1],
                  ' train_acc: ', train_acc_history[-1])

            print('valid_loss: ', validation_loss_history[-1],
                  ' valid_acc: ', validation_acc_history[-1])

    def evaluation_run(self, display_most_unsure: int=10, detailed_view=False):
        assert self._session

        # I run through the validation dataset to create a set of statistics about the trained model
        self._valid_batch_generator.reset()

        # 1. Remember the n correct results with lowest probability ( n = validation_save_worst_n_correct )
        # 2. Remember all incorrect classifications
        worst_n_correct_predictions = {}
        incorrect_predictions = {}

        for step in range(self._steps_per_epoch_valid):
            # Get the batch from training batch generator
            # This will yield exactly 1 Sample
            batch_x, batch_y = self._valid_batch_generator.next()

            # Prepare the graph nodes to run and the graph inputs
            ops_to_run = [self._tf_graph_elements['evaluate']]

            feed_dict = {self._tf_graph_elements['x']: batch_x,
                         self._tf_graph_elements['y']: batch_y,
                         self._tf_graph_elements['dropout']: 1.0}

            # Just get the loss and accuracy over this batch
            eval_results = self._session.run(ops_to_run, feed_dict=feed_dict)

            # Iterate through the batch of results
            for eval_hits, eval_probs in eval_results:
                for hit_idx, eval_hit in enumerate(eval_hits):
                    hit = eval_hit
                    probabilities = eval_probs[hit_idx]
                    prob_of_correct_label = probabilities[batch_y[hit_idx]]

                    if detailed_view:
                        predictions_dict = {}
                        for label_idx, pred in enumerate(eval_probs[hit_idx]):
                            predictions_dict.update({self._class_translations[label_idx]: pred})

                        util.visualize_single_prediction(img=batch_x[hit_idx],
                                                         title=self._class_translations[batch_y[hit_idx]],
                                                         predictions=predictions_dict)

                    if not hit:
                        incorrect_predictions.update(
                            {len(incorrect_predictions): [batch_x[hit_idx],
                                                          self._class_translations[batch_y[hit_idx]],
                                                          prob_of_correct_label]})

                    else:
                        # If the model hit the correct class,
                        # remember the probability and update the worst_n_correct_preds
                        max_prob_k = None
                        max_prob = 0.0
                        for k in worst_n_correct_predictions:
                            if worst_n_correct_predictions[k][2] > max_prob:
                                max_prob = worst_n_correct_predictions[k][2]
                                max_prob_k = k

                        insert_item = False
                        if max_prob_k is None or len(worst_n_correct_predictions) < display_most_unsure:
                            max_prob_k = len(worst_n_correct_predictions)
                            insert_item = True

                        insert_item = insert_item or (max_prob > prob_of_correct_label)

                        if insert_item:
                            worst_n_correct_predictions.update(
                                {max_prob_k: [batch_x[hit_idx],
                                              self._class_translations[batch_y[hit_idx]],
                                              prob_of_correct_label]})

        print('Evaluation accuracy : ', str((1.0-(len(incorrect_predictions)  / len(self._validation_set[0])))* 100.0), ' %')

        util.visualize_predictions(worst_n_correct_predictions, 'Correct predictions with lowest probabilities')
        util.visualize_predictions(incorrect_predictions, 'Incorrect predictions')

    def print_dataset_statistics(self, dataset_name: str='train'):
        if dataset_name not in ['train', 'val']:
            raise Exception('Unrecognized dataset name! Try train or val.')

        dataset = self._training_set if dataset_name == 'train' else self._validation_set
        util.print_datasets_stats(dataset[0], dataset[1])

    def visualize_dataset_images(self, dataset_name: str='train', samples_per_class: int=5, max_classes: int=5):
        if dataset_name not in ['train', 'val']:
            raise Exception('Unrecognized dataset name! Try train or val.')

        dataset = self._training_set if dataset_name == 'train' else self._validation_set

        util.visualize_dataset_content(dataset[0],
                                       dataset[1],
                                       self._class_translations,
                                       n_samples=samples_per_class,
                                       n_classes=max_classes)

    def visualize_dataset_frequencies(self, dataset_name: str='train'):
        if dataset_name not in ['train', 'val']:
            raise Exception('Unrecognized dataset name! Try train or val.')

        dataset = self._training_set if dataset_name == 'train' else self._validation_set

        util.visualize_dataset_frequencies(dataset[1],
                                           self._class_translations)

    def visualize_image_augmentation(self, k_samples: int=5, k_augmentations: int=9):
        # Sample k training-samples, augment them and display the results
        augment_x, augment_y = skutil.resample(self._training_set[0],
                                               self._training_set[1],
                                               n_samples=k_samples)

        for k_s in range(k_samples):
            augmentations = {}

            # Push in the original
            augmentations.update({len(augmentations): [self._preprocessing_function(augment_x[k_s]),
                                                       self._class_translations[augment_y[k_s]],
                                                       0.0]})

            for k_a in range(k_augmentations):
                augmentations.update({len(augmentations): [self._preprocessing_function(
                    self._augmentation_function(augment_x[k_s])),
                    self._class_translations[augment_y[k_s]],
                    0.0]})

            util.visualize_predictions(augmentations, 'Test Augmentations')


# Train a model
# ---------------------------------------------------------------------------------------------------------

# Instantiate the ModelTrainer !
trainer = ModelTrainer(log_directory='logs',
                       path_to_traindata='datasets/train.p',
                       path_to_valdata='datasets/valid.p',
                       translations_file='signnames.csv')

# The preprocessing function will be a lambda function that calls some functions of the image_preprocessor
trainer.set_preprocessing_function((lambda x: trainer.image_preprocessor.normalize_center(
    trainer.image_preprocessor.apply_clahe(x))))

# The augmentation function is also a lambda function that uses the internal image augmenter of the ModelTrainer
trainer.set_augmentation_function((lambda x: trainer.image_augmenter.process(x)))

# Set the parameter of this training process.
trainer.set_training_parameter(learning_rate=1e-3, batch_size=128, epochs=50, optimizer='adam')

# Print out some information about the datasets
print('Training set statistics:')
trainer.print_dataset_statistics(dataset_name='train')

print('Validation set statistics:')
trainer.print_dataset_statistics(dataset_name='val')

# Print frequencies
print('Visualization of training class frequencies:')
trainer.visualize_dataset_frequencies(dataset_name='train')

# Print frequencies after equalisation
trainer.equalize_traindata_class_frequencies()
print('Visualization of training class frequencies after equalisation:')
trainer.visualize_dataset_frequencies(dataset_name='train')

# Print frequencies
print('Visualization of validation class frequencies:')
trainer.visualize_dataset_frequencies(dataset_name='val')

# Visualize some images
print('Visualization of some images of some classes:')
trainer.visualize_dataset_images(dataset_name='train')

# Visualize Augmentation
print('Visualization image augmentation:')
trainer.visualize_image_augmentation()

# Get the shape of the training images
input_shape = trainer.get_image_shape()

# Get the count of classes from the ModelTrainer
num_classes = trainer.get_num_classes()

# Define the model to train
custom_squeezenet = models.TfCustomSqueezeNet(input_shape=[None,
                                                           input_shape[0],
                                                           input_shape[1],
                                                           input_shape[2]],
                                              n_classes=num_classes,
                                              kernel_regularization=0.01,
                                              dropout_training_value=0.5)
# Set the model to the ModelTrainer
trainer.set_model(custom_squeezenet)

# Fit the model with the specified optimizer to the training set
# trainer.fit(experiment_name='final_experiment')

# Evaluate on the validation set
# trainer.evaluation_run(display_most_unsure=10, detailed_view=False)

# Get the last (best) Checkpoint filename
#last_checkpoint = trainer.get_last_checkpoint_filename()

last_checkpoint = 'logs/final_experiment/checkpt-0.0139639638592-39'


# Clear the Session Graph
tf.reset_default_graph()


# Load a model and evaluate the test-set
# ---------------------------------------------------------------------------------------------------------

# Instantiate the ModelTrainer !
trainer = ModelTrainer(log_directory='logs',
                       path_to_traindata='datasets/train.p',
                       path_to_valdata='datasets/test.p',
                       translations_file='signnames.csv')

# The preprocessing function will be a lambda function that calls some functions of the image_preprocessor
trainer.set_preprocessing_function((lambda x: trainer.image_preprocessor.normalize_center(
    trainer.image_preprocessor.apply_clahe(x))))

# The augmentation function is also a lambda function that uses the internal image augmenter of the ModelTrainer
trainer.set_augmentation_function((lambda x: trainer.image_augmenter.process(x)))

# Set the parameter of this training process.
trainer.set_training_parameter(learning_rate=1e-3, batch_size=128, epochs=50, optimizer='adam')

# Get the shape of the training images
input_shape = trainer.get_image_shape()

# Get the count of classes from the ModelTrainer
num_classes = trainer.get_num_classes()

# Define the model to train
custom_squeezenet = models.TfCustomSqueezeNet(input_shape=[None,
                                                           input_shape[0],
                                                           input_shape[1],
                                                           input_shape[2]],
                                              n_classes=num_classes,
                                              kernel_regularization=0.01,
                                              dropout_training_value=0.5)
# Set the model to the ModelTrainer
trainer.set_model(custom_squeezenet)

# Load the last checkpoint from previous training
trainer.load_weights(last_checkpoint)

# Evaluate on the validation set
trainer.evaluation_run(display_most_unsure=10, detailed_view=False)

# Clear the Session Graph
tf.reset_default_graph()


# Load a model and evaluate the custom test-set
# ---------------------------------------------------------------------------------------------------------

# Instantiate the ModelTrainer !
trainer = ModelTrainer(log_directory='logs',
                       path_to_traindata='datasets/train.p',
                       path_to_valdata='datasets/custom_test.p',
                       translations_file='signnames.csv')

# The preprocessing function will be a lambda function that calls some functions of the image_preprocessor
trainer.set_preprocessing_function((lambda x: trainer.image_preprocessor.normalize_center(
    trainer.image_preprocessor.apply_clahe(x))))

# The augmentation function is also a lambda function that uses the internal image augmenter of the ModelTrainer
trainer.set_augmentation_function((lambda x: trainer.image_augmenter.process(x)))

# Set the parameter of this training process.
trainer.set_training_parameter(learning_rate=1e-3, batch_size=128, epochs=50, optimizer='adam')

# Get the shape of the training images
input_shape = trainer.get_image_shape()

# Get the count of classes from the ModelTrainer
num_classes = trainer.get_num_classes()

# Define the model to train
custom_squeezenet = models.TfCustomSqueezeNet(input_shape=[None,
                                                           input_shape[0],
                                                           input_shape[1],
                                                           input_shape[2]],
                                              n_classes=num_classes,
                                              kernel_regularization=0.01,
                                              dropout_training_value=0.5)
# Set the model to the ModelTrainer
trainer.set_model(custom_squeezenet)

# Load the last checkpoint from previous training
trainer.load_weights(last_checkpoint)

# Evaluate on the validation set
trainer.evaluation_run(display_most_unsure=10, detailed_view=True)

# Clear the Session Graph
tf.reset_default_graph()

