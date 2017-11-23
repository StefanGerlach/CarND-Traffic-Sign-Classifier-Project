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

  * **packages\dataset_utils.py** this is my collection of dataset utilities
  * **packages\tf_models.py** a wrapper/template for my tensorflow models
  * **packages\tf_train_utils.py** here live the loss functions, optimizer definitions, batch generator, class frequency equalizers, data augmenter and image preprocessors
  * **train_evaluate_trafficsign_clsf.py** - here I definied my ModelTrainer class and do actual training, visualization and evaluation
  * **Traffic_Sign_Classifier.ipynb** in this notebook I collect all my code in one file for better overview of code and the outputs I generated
  * **Traffic_Sign_Classifier.html** the notebook as html
  
  

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
To compensate the copies of the images in the training set and to give the dataset more variation for better network generalization and less overfitting, I apply image augmentation.
