## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this writeup I want to explain my approach of a traffic sign classifier based on deep learning architectures. During this project I am going to train and validate a model that is capable of classifying images of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and images from the traffic signs of Lübeck, Germany. 

### Part 1
#### At first I am going to cover the following points about the dataset and image processing:

  * Dataset Visualization
  * Image Preprocessing
  * Class distribution normalization
  * Image Batching to feed them into the classifier during training
  * Image Augmentation
  
### Part 2
#### Then I summarize what deep learning architecture I chose and how I monitored training and validation:

  * Tensorboard for monitoring
  * Testing LeNet as a baseline
  * Tweaking LeNet
  * Using a custom model inspired by [SqueezeNet](https://arxiv.org/abs/1602.07360)
  
### Part 3 
#### Finally I present my results and reflect them

  * Evaluate the Test-Set 
  * Evaluate completely new images
  * Reflection



## Directory Overview
I want to describe my directory structure first, to give any reader a short overview what to find in my fork of the Udacity-Traffic-Sign-Classifier-Project repository.

| File | Description |
| ---- | ----------- |
| **packages\dataset_utils.py** | This is my collection of dataset utilities |
| **packages\tf_models.py** | A wrapper/template for my tensorflow models |
| **packages\tf_train_utils.py** | Here live the loss functions, optimizer definitions, batch generator, class frequency equalizers, data augmenter and image preprocessors |
| **train_evaluate_trafficsign_clsf.py** | Here I definied my ModelTrainer class and do actual training, visualization and evaluation |
| **Traffic_Sign_Classifier.ipynb** | Un this notebook I collect all my code in one file for better overview of code and the outputs I generated |
| **Traffic_Sign_Classifier.html** | The notebook as html |
  
  

## Part 1
#### Data Visualization
---
First I analyzed the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) in basic matter: counting and displaying the classes and images. The splits are already done by Udacity and here are the statistics about the datasets:

| Dataset | Number of samples |
| :-----: | :---------------: |
| Training | 34799 |
| Validation | 4410 |
| Test | 12630 |

There are 43 unique classes of traffic signs in this dataset. To have a visual impression of these 32x32 RGB images, I created a print function to display some samples of all/some classes. Here are some (upscaled and interpolated!) images of some classes:

![animal_crossing](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/animal_crossing_example_images.PNG "Examples for animal crossing traffic signs")
![end_no_passing](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/end_of_no_passing_images_example.PNG "Examples for end of no passing traffic signs")
![slippery_road](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/slippery_road_images_example.PNG "Examples for slippery road traffic signs")


#### Image Preprocessing
---
For this project I decided to stick with RGB images since the color of a traffic sign can make a significant difference. For example the two german traffic signs for end of speed limitation and end of minimum speed:

![end_max_speed](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/ende_zulaessige_geschwindigkeit_30.png "Example for end of speed limit")

![min_speed](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/ende_mindest_geschwindigkeit_30.png "Example for minimum speed")

Having the latter trafic sign in grayscale will not fool a human, but maybe a convnet.

![min_speed](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/ende_mindest_geschwindigkeit_30_gray.png "Example for minimum speed")

So the blue color is a significant information in this case, which I don't want to drop. 

In order to enhance contrast I used the contrast limited histogram equalisation function of opencv. This enhances contrast and edges a little bit. Before using the images for training a classifier I normalize them with the basic function x: (x - 128.) / 128. This creates images in a range of -1 to 1 of type float32 .


#### Class distribution normalization
---
What I additionally did was analyzing the distribution of class-occurences. I plotted a histogram of class occurences in the training-set:
![hist_train](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/hist_traindata.png "histogram of class occurences in trainset")

To avoid bias in the classifier, I normalize the frequencies by random sampling images of a specific class and copying them into the set until I reach an equal count of images for each class. After that operation the histogram looks like this:
![hist_train_equal](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/hist_testdata_equalized.png "histogram of normalized class occurences in trainset")

I dont worry about multiples images in a class because I am going to
  * Random sample a batch from the training-set
  * Apply data augmentation


#### Image batching to feed them into the classifier during training
---
For the training process of the deep neural network I want to randomly select a batch of images at each training step. For this purpose I created a BatchGenerator that does the job. The batch generator additionally has the image preprocessing and image augmentation function that is going to be applied on each batch before it is fed to the network. The complete trainingset is shuffled at the start of a new epoch.


#### Image Augmentation
---
To compensate the copies of the images in the training set and to give the dataset more variation for better network generalization and less overfitting, I apply image augmentation. To do this, I use the keras ImageGenerator to apply 

 * Image rotation in a range of -20 to +20 degrees
 * Image translation in a range of -10 to +10 %
 * Image Zooming in range of -20 to +20 %
 * Intensity shift 
 * Image Shearing

To visualize, how aggressive the augmentation is, I use a visualization function where the most left image is the original and all other images are slight variations of it:

![augment_1](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/augmentation_0.PNG "Testing image augmentation")
![augment_2](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/augmentation_1.PNG "Testing image augmentation")



## Part 2
#### Tensorboard for monitoring
---
To monitor the training process I use tensorboard, which comes with the tensorflow-package. This extremely handy tool displays all information I need for this small project in the browser. To create statistics for tensorboard, a summary writer is instantiated and gets a scalar summary operation for the training loss, training accuracy, validation loss and validation accuracy. The Tensowflow-Graph is automatically shown in tensorboard!

For saving my checkpoints I encapsulated the tensorflow.train.saver in my TrainSaver class. This class saves a new checkpoint if the validation loss decreases with the naming convention < logdir >checkpt-< val_loss >-< epoch >.
 
![tensorboard_0](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/tensorboard_summaries.png "Tensorboard")


#### Testing LeNet as a baseline
---

**TFModel** is the class I created to have an easy interface for creating new deep learning model architectures. Here I wrote some functions that wrap tensorflow functions like conv2d, fc_layer or dropout. My wrapper functions feel like a bit like Keras, that inspired me in doing this. 

I took the LeNet architecture from the Udacity lab as a baseline.

But before doing this, to put it all together I created a ModelTrainer class. Here all code comes together. The ModelTrainer has information about 
  * Log directory
  * Paths to datasets and the class translations
  * The class that normalizes the class occurences
  * The class that can preprocess images
  * The class that is able to augment images
  * The batch generator
  * The training optimizer
  * All training hyperparameters
  * The deep learning model 
  * The tensorflow session and graph operations
  * The model saver
  
After instantiation of the ModelTrainer with the directories of the datasets (train, validation), the class translations and the log directory there are just the following functions to call:

  * set_preprocessing_function()
  * set_augmentation_function()
  * set_training_parameter()
  * set_model()
  * fit()
  * evaluation_run()


When all of these functions have executed, I take a look into Tensorboard and reflect how well the model was trained during this process. Most of the time I used these hyperparameters and settings:

| Parameter | Value |
| :-----: | :---------------: |
| Base Learning Rate | 1e-3 |
| Optimizer | Adam (RMS with momentum) |
| Batch Size | 128 |
| Epochs | 50 |
| Kernel Regularization | 1e-2 | 
| Dropout (chance to drop) | 0.25 - 0.5 |
| Loss Function | softmax cross entropy with logits |

I decided for Adam because it has not much parameters to adjust but the base learning rate. It is very adaptive, but nevertheless I expect good results with SGD(lr = 1e-4, momentum = 0.9, + learning rate decay), too. 

The Kernel Regularization factor of 1e-2 is very high because I want the network not to overfit on this small dataset and have the convolutional weights sparse.

Dropout serves as method for regularization. The count of epochs ( 50 ) is chosen empirically.


#### Tweaking LeNet
---

The first results did look like in the next screenshot of Tensorboard. The baseline model (in blue) performed quite well - I readhed a validation accuracy of about 92,3 %. But when I had a look over the validation loss, there was a slight tendency of increasement / overfitting to the dataset. So I introduced a dropout of 0.25 (chance to drop) and observed, that the val_loss decreased even more and the val_acc reached 93,3 %!

![tensorboard_1](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/lenet_baseline_tb.png "Tensorboard")


#### Using a custom model inspired by [SqueezeNet](https://arxiv.org/abs/1602.07360)
---

For the next step I wanted to implement a new architecture that has some interesting layer architectures and is not as big as for example Inception V3, ResNet or VGGx. I like the idea to have a great performance with less parameter and this is where SqueezeNet shines. So I decided to implement the FireModules from the SqueezeNet Paper and put them into a smaller network. 

So my custom architecture can be described like

##### SqueezeBlock:

| Layer | Description |
| :-----: | :---------------: |
| 1 | input -> conv2d 1x1 relu activation |
| 2 | 1 -> conv2d 1x1 relu activation |
| 3 | 1 -> conv2d 3x3 relu activation |
| 4 | 2, 3 -> concatenation |

##### Custom SqueezeNet-inspired Network

| Layer | Description | Output Shape (Batch x Height x Width x Channels) |
| :---: | :---------: | :----------: |
| 0 | input | 128 x 32 x 32 x 3 |
| 1 | input -> conv2d 3x3 relu activation | 128 x 32 x 32 x 24 | 
| 2 | 1 -> maxpool 2x2 | 128 x 16 x 16 x 24 |
| 3 | 2 -> squeezeblock | 128 x 16 x 16 x 48 |
| 4 | 3 -> maxpool 2x2 | 128 x 8 x 8 x 48 |
| 5 | 4 -> squeezeblock | 128 x 8 x 8 x 96 |
| 6 | 5 -> squeezeblock | 128 x 8 x 8 x 96 |
| 7 | 6 -> dropout |  128 x 8 x 8 x 96 |
| 8 | 7 -> maxpool 2x2 | 128 x 4 x 4 x 96 |
| 9 | 8 -> conv2d 3x3 relu activation | 128 x 4 x 4 x 64 |
| 10 | 9 -> flatten | 128 x 1024 |
| 11 | 10 -> dropout | 128 x 1024 |
| 12 | 11 -> fc 64 neurons | 128 x 64 |
| 13 | 12 -> dropout | 128 x 64 |
| 14 | 13 -> fc 43 neurons (num classes) | 128 x 43 |


My first experiments dramatically failed with my custom SqueezeNet. And this is where I discovered the real importance of weight initialization! I switched to Xavier initialization and the network converged.

[Understanding the difficulty of training deep feedforward neural networks, Xavier Glorot, Yoshua Bengio](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf)


##### Experiments
---

| Curve color | Experiment Description |
| :----- | :--------------- |
| Blue | LeNet Baseline |
| Red | tweaked LeNet |
| Light Blue | The first version of custom SqueezeNet with Dropout at 0.5 |
| Green | The final experiment with custom SqueezeNet, Dropout 0.5 and Image Augmentation |


![tensorboard_2](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/lenet_squeeze_tb.png "Tensorboard SqueezeNet")



## Part 3
#### Evaluate the Test-Set 
---

After having finished all tweaking I run the test-set. These are my results:

| **Dataset** | **Accuracy** |
| :-----: | :------: |
| **Validation** | **99.71 %** |
| **Test** | **98.53 %** | 



##### Correct predicted images of the test-set but lowest probability

![low_prob_testset](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/low_prob_testset.png "Low probability on testset")


##### Some of the mis-classifications of the test-set

![wrong_testset](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/incorrect_test.png "failures on testset")

Most of the misclassifications are blurred or occluded images of traffic signs, but there are also some clearly readable images (limit 60 km/h). Maybe the structured background misled the classifier. More data augmentation could help to tackle this problem.


#### Evaluate completely new images
---

I did take some new images around Lübeck, Germany with my camera and cropped them to the size of 32x32 pixels. I selected 6 clearly readable images and 6 with bad quality. So my expectation was an around 50 % accuracy of the system. To generate a new dataset, I used my function create_dataset(). Then I could run the evaluation. 

I achieved 8/12 correct classifications. Here are the correct predicted images:

![correct_customset](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/correct_custom.png "Correct predicted custom images")

For the incorrect predictions I want to display these images in detail with their specific probability after softmax:

![incorrect_custom_1](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/detail_1.png "1 incorrect custom images")

![incorrect_custom_2](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/detail_5.png "2 incorrect custom images")

![incorrect_custom_3](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/detail_7.png "3 incorrect custom images")

![incorrect_custom_4](https://github.com/StefanGerlach/CarND-Traffic-Sign-Classifier-Project/blob/master/images/detail_8.png "4 incorrect custom images")



#### Reflection
---

This is an interesting project that has a lot of potential. Further tweaking the model, like making it deeper with more layers or more kernels with additional data augmentation methods and regularization methods to not let the network overfit on this small dataset could get the final percentages on this dataset. 

But: this is a quite small dataset! Transfer learning could help to use a pre-trained network and finetune on this tiny dataset. Or even some other dataset (if available) could be used to extend the amount of training images.

When looking forward, the network could even be translated in a fully convolutional network to find the traffic signs in a larger image. The current network only works on patches where the traffic sign **acutally is**. The Tensorflow-API allows flexible construction of deep learning models, so this step wouldn't even take a lot of work.


Looking forward to the next project of this Nanodegree.
