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
