# **Traffic Sign Recognition Writeup**

---

[//]: # (Image References)

[bar1]: resources/bar1.png "Dataset Graph"
[bar2]: resources/bar2.png "Augmented vs Original Dataset"
[1]: resources/1.png "Img/Label Visualization"
[2]: resources/2.png "Img/Label Visualization"
[3]: resources/3.png "Img/Label Visualization"
[lab]: resources/lab.png "LAB colorspace"
[prep1]: resources/og.png "Original Image"
[prep2]: resources/prep.png "Preprocessed Image"
[aug]: resources/aug.png "Augmented Image"
[cust1]: resources/cust.png "Custom Images"
[cust2]: resources/cust_prep.png "Preprocessed Custom Images"
[topk1]: resources/topk1.png "Top 5 Predictions(linear scale)"
[topk2]: resources/topk2.png "Top 5 Predictions(linear scale)"
[logk1]: resources/logk1.png "Top 5 Predictions(log scale)"
[logk2]: resources/logk2.png "Top 5 Predictions(log scale)"
[vis1]: resources/vis.png "Filter Visualization"

---

You're reading it! and here is a link to my [project code](https://github.com/kemfic1/Traffic-Sign-Classifier/blob/master/writeup.ipynb)

#### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The bar graph displays the distribution of image examples within each class in our training set. As you can see, the distribution of examples per class is very uneven, which can result in the network to have a predisposition to choose certain classes over others when it is uncertain of the class of an image.

![alt text][bar1]

Below you can view examples of each dataset, along with their labels

![alt_text][1]
![alt_text][2]
![alt_text][3]


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

##### Model Architecture

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


* validation set accuracy of 0.945
* test set accuracy of 0.935

I picked Yann Lecun's original LeNet as I wasn't too well-versed with convolutional networks, and wanted to experiment with a simple network that could pass the accuracy requirement. I though this network would work well as convolutional networks are very useful in image processing applications. A 93.5% test accuracy shows that the model is able to generalize the features in different traffic signs relatively well, but I am not too satisfied with the results.

A much more promising architecture for this application would probably be Capsule Networks. I recently read Geoff Hinton's paper *Dynamic Routing Between Capsules*, and I believe this network would work *much* better for traffic sign classification. Since the images of traffic signs within our datasets look very similar, and are taked from various angles, capsule networks would perform very well. Capsule networks are able to generalize images of objects taken from different perspectives very well, as they can detect not only specific features, but also the spatial relationships between the features in an object. Hinton states this in his paper,"Capsules use neural activities that vary as viewpoint varies rather than trying to eliminate viewpoint variation from the activities: ... They can deal with multiple different affine transformations of different objects or object parts at the same time." I'm currently working on replicating the Capsule Network, and applying it to the traffic sign dataset to analyze its performance on different data. I'm excited to see the results in a few days.


### Test a Model on New Images

Here are some German traffic signs that I screenshotted and cropped from Google Maps, along with the top 5 predictions for its class:

![alt text][cust1]
![alt_text][cust2]

![alt_text][topk1]
![alt text][topk2]


Surprisingly enough, the model classified the data with 100% accuracy, and with very high confidence for each image. Below you can see the top 5 predictions, and prediction confidences, but on the log scale.

![alt_text][logk1]
![alt text][logk2]

This might be a sign that the model is overfitting, as the other predictions have a very low confidence value.

BONUS: here's a cool vis. Notice how certain parts of the image get different intensities of activation in different nodes.
![alt_text][vis1]

