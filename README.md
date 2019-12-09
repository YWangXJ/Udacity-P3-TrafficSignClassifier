# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.PNG "Visualization"
[image2]: ./examples/original.PNG "Original"
[image3]: ./examples/grayscale.PNG "Grayscaling"


[image4]: ./examples/12504.jpg "Traffic Sign 1"
[image5]: ./examples/12523.jpg "Traffic Sign 2"
[image6]: ./examples/12541.jpg "Traffic Sign 3"
[image7]: ./examples/12549.jpg "Traffic Sign 4"
[image8]: ./examples/12562.jpg "Traffic Sign 5"

[image9]: ./examples/FeatueMaps.PNG "Featue Maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute on different labels

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step of the preprocessing, I normalized the image data for better learning performance (similar to feature scaling)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer        		|     Description	   					| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							| 
| Convolution1     	  | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling		   | 2x2 stride,  outputs 16x16x64 				|
| Convolution2 	      | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling		   | 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| inputs 400 outputs 120 						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 120 outputs 84 						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 84 outputs 10							|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the Adam optimize, batch size 128 and EPOCHS 10 and learning rate 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final validation set accuracy is 0.940


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I directly use the example from previous Lab code. the LeNet architecture
* What were some problems with the initial architecture?
The final validation accuracy is relatively low. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added dropout after every fully connected layers and apply 0.5 keep_probability of the dropout. The training results are running faster and having better accuracy on validation set. 
* Which parameters were tuned? How were they adjusted and why?
1) tried to tune epochs to 20 but found there is no significant performance.So I changed back to 10.
2) tried to add another convolution layer between the fisrt and second convolution layer, but the accuracy significantly decreased.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The dropout layer helps a lot since it successfully recude the overfitting problem and significantly increase the trainning efficiency.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet architecture was chosen

* Why did you believe it would be relevant to the traffic sign application?
LeNet was designed for the hand writing recognition. The traffic sign can be considered as a printed markings similar to the hand writing.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Overall, the accuracy is good. all larger than 90%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because it has a obvious shade on the arrow.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                  |     Prediction	     					| 
|:---------------------:           |:---------------------------------------------:| 
|Speed limit (50km/h)              | Speed limit (50km/h)						| 
|Ahead only    			           | Ahead only								|
|Road narrows on the right	         | Road narrows on the right	 				|
|Right-of-way at the next intersection	| Right-of-way at the next intersection			|
|Ahead only  			           | Ahead only      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the validation set 94%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the second image, the model is relatively sure that this is a Ahead Only (probability of 99.89%), and the image is a Ahead Only stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		  | 
|:---------------------:|:---------------------------------------------:| 
| 98.89%       			| Ahead Only   								| 
| .05%     				| No passing 								  |
| .04%					| No passing for vehicles over 3.5 metric tons	|
| .01%	      			| Speed limit (60km/h)					 	  |
| .01%				    | Turn left ahead      						 |


For the first image

| Probability         	|     Prediction	        		  | 
|:---------------------:|:---------------------------------------------:| 
| 100%       			| Speed limit (50km/h)   					| 


For the third image
| Probability         	|     Prediction	        		  | 
|:---------------------:|:---------------------------------------------:| 
| 98.47%       			| Road narrows on the right   				| 
| .49%     				| Pedestrians 								  |
| .20%					  | Bicycles crossing                  	|
| .20%	      			| Traffic signals   					 	  |
| .18%				    | General caution      						 |

For the fourth image
| Probability         	|     Prediction	        		  | 
|:---------------------:|:---------------------------------------------:| 
| 100%       			| Right-of-way at the next intersection 	| 

For the fifth image
| Probability         	|     Prediction	        		  | 
|:---------------------:|:---------------------------------------------:| 
| 100%       			| Ahead Only   								| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image9]


As shown in the figure, the first  layer mostly use the sign shape; the second  layer consider both sign shape and the number ; the third and fourth layer considers more detailed characteristics of the image.
