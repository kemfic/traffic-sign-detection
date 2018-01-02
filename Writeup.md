# **Traffic Sign Recognition Writeup**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar1]: Writeup_Assets/bar1.png "Dataset Graph"
[bar2]: Writeup_Assets/bar2.png "Augmented vs Original Dataset"
[1]: Writeup_Assets/1.png "Img/Label Visualization"
[2]: Writeup_Assets/2.png "Img/Label Visualization"
[3]: Writeup_Assets/3.png "Img/Label Visualization"
[lab]: Writeup_Assets/lab.png "LAB colorspace"
[prep1]: Writeup_Assets/og.png "Original Image"
[prep2]: Writeup_Assets/prep.png "Preprocessed Image"
[aug]: Writeup_Assets/aug.png "Augmented Image"
[cust1]: Writeup_Assets/cust.png "Custom Images"
[cust2]: Writeup_Assets/cust_prep.png "Preprocessed Custom Images"
[topk1]: Writeup_Assets/topk1.png "Top 5 Predictions(linear scale)"
[topk2]: Writeup_Assets/topk2.png "Top 5 Predictions(linear scale)"
[logk1]: Writeup_Assets/logk1.png "Top 5 Predictions(log scale)"
[logk2]: Writeup_Assets/logk2.png "Top 5 Predictions(log scale)"
[vis1]: Writeup_Assets/vis.png "Filter Visualization"

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kemfic1/Traffic-Sign-Classifier/blob/master/writeup.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The bar graph displays the distribution of image examples within each class in our training set. As you can see, the distribution of examples per class is very uneven, which can result in the network to have a predisposition to choose certain classes over others when it is uncertain of the class of an image.

![alt text][bar1]

Below you can view examples of each dataset, along with their labels

![alt_text][1]
![alt_text][2]
![alt_text][3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing
As a first step, I applied an adaptive histogram equalization algorithm to normalize our images, and then converted the colorspace of the image from RGB to LAB. 

![alt_text][lab]

It stores color data into a 3 dimensional vector, where lightness(L), red-green(A), and blue-green(B) are independent of each other, similar to HSV. I had also tested out the HSV colorspace, but my model seemed to generalize better with the LAB colorspace. 

Here is an example of the images before/after the preprocessing steps.

![alt text][prep1]
![alt text][prep2]

##### Data Augmentation

As you could see with the previous bar graph, the class examples are distributed very unevenly. Some classes have over 200 examples, while others have as little as 250 examples. This is not good for training a network, as the classes with fewer examples can be overfit by the network, and can cause the network to have a predisposition towards incorrectly classifying images towards other classes.

To prevent this from happening, we need to have a more uniform dataset. I decided to augment the images by using skimage.transform.AffineTansform() and randomly rotating, scaling, and translating images until enough examples are augmented for all classes to have the same number of examples. 

The bar graph below displays the distribution of examples before and after the augmentation process. The orange bars show the original dataset, and the blue bars represent the augmented dataset. 

![alt_text][bar2]

Here is an example of an augmented image.

![alt_text][aug]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| Leaky ReLU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Leaky ReLU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten       | input: 5x5x16, output: 650    |
| Fully connected		| Input: 650, output: 120        									|
| Leaky ReLU    |     |
| Fully Connected    | Input: 120 Output: 84        |
| Leaky ReLU   |      |
| Fully Connected    | Input: 84, Output: 10 |
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.945
* test set accuracy of 0.935

I picked Yann Lecun's original LeNet as I wasn't too well-versed with convolutional networks, and wanted to experiment with a simple network that could pass the accuracy requirement. I though this network would work well as convolutional networks are very useful in image processing applications. A 93.5% test accuracy shows that the model is able to generalize the features in different traffic signs relatively well, but I am not too satisfied with the results.

A much more promising architecture for this application would probably be Capsule Networks. I recently read Geoff Hinton's paper *Dynamic Routing Between Capsules*, and I believe this network would work *much* better for traffic sign classification. Since the images of traffic signs within our datasets look very similar, and are taked from various angles, capsule networks would perform very well. Capsule networks are able to generalize images of objects taken from different perspectives very well, as they can detect not only specific features, but also the spatial relationships between the features in an object. Hinton states this in his paper,"Capsules use neural activities that vary as viewpoint varies rather than trying to eliminate viewpoint variation from the activities: ... They can deal with multiple different affine transformations of different objects or object parts at the same time." I'm currently working on replicating the Capsule Network, and applying it to the traffic sign dataset to analyze its performance on different data. I'm excited to see the results in a few days.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I screenshotted and cropped from Google Maps, along with the top 5 predictions for its class:

![alt text][cust1]
![alt_text][cust2]

![alt_text][topk1]
![alt text][topk2]


Surprisingly enough, the model classified the data with 100% accuracy, and with very high confidence for each image. Below you can see the top 5 predictions, and prediction confidences, but on the log scale.

![alt_text][logk1]
![alt text][logk2]

This might be a sign that the model is overfitting, as the other predictions have a very low confidence value. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

View previous answer.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

View previous answer.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt_text][vis1]

The image here shows the outputs for the first convolution layer. From what I understand, Filters 9, 12, and 0 seem to "detect" the background of the sign. Filters 5, 7, 10, 11, and 15 output higher values when they "detect" certain diagonal lines within the sign. The backgrounds of the images dont seem to activate the filters as much as certain features of the image, which makes sense, as the network is trained to detect useful features within images that allow it to classify images.



