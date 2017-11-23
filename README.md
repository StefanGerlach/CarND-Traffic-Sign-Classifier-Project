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
  * Image batching to feed them into the classifier during training
  
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

### Directory Overview
---
I want to describe my directory structure first, to give any reader a short overview what to find in my fork of the Udacity-Traffic-Sign-Classifier-Project repository.

  * **packages\dataset_utils.py** this is my collection of dataset utilities
  * **packages\tf_models.py** a wrapper/template for my tensorflow models
  * **packages\tf_train_utils.py** here live the loss functions, optimizer definitions, batch generator, class frequency equalizers, data augmenter and image preprocessors
  * **train_evaluate_trafficsign_clsf.py** - here I definied my ModelTrainer class and do actual training, visualization and evaluation
  * **Traffic_Sign_Classifier.ipynb** in this notebook I collect all my code in one file for better overview of code and the outputs I generated
  * **Traffic_Sign_Classifier.html** the notebook as html
  

### Part 1
---
#### Data Visualization

First I analyzed the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) in basic matter: counting and displaying the classes and images. The splits are already done by Udacity and here are the statistics about the datasets:

| Dataset | Number of samples |
| :-----: | :---------------: |
| Training | 34799 |
| Validation | 4410 |
| Test | 12630 |

There are 43 unique classes of traffic signs in this dataset. To have a visual impression of these 32x32 RGB images, I created a print function to display some samples of all/some classes. Here are some (upscaled and interpolated!) images of some classes:

[animal_crossing]: images/animal_crossing_example_images.PNG "Examples for animal crossing traffic signs"

