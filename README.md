
## Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1a]: ./examples/car.jpeg
[image1b]: ./examples/not_car.jpeg
[image4a]: ./examples/bb1.jpg
[image4b]: ./examples/bb2.jpg
[image4c]: ./examples/bb3.jpg
[image5]: ./examples/heatmap.jpg
[image6]: ./examples/bb_final.jpg
[video1]: ./test_video_output/project_video.mp4

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for this step is contained in the file called `search_classify.py` and some other file that I have lost by accident.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![vehicle][image1a]
![non-vehicletext][image1b]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#### 2. Final choice of HOG parameters.

I have tried the following parameters:
- Colorspaces RGB, HSV, LUV, HLS, YUV, YCrCb
- Each channel for all colorspaces and all channels combined
- Orientations between 2 and 15
- Pixel per cell between 4 and 16
- Cell per block between 2 and 4

Unfortunatly I have lost the training file and the logs files that shows all the results.

There were more combinations that gave accuracy >= 0.99 but I have chosen the following combination to continue with:
YCrCb, all channel, orientations of 9, 8 pixels per cell and 2 cells per blocks.

Spacial features and histogram features didn't improve the results so I didn't used them.

I trained a linear SVM using kernel rbf with grid search and tried the C values:
0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10

A C value of 1 gave the best result.

### Sliding Window Search

#### 1. Sliding window search.
  
The sliding window search is implemented in the `find_car.py` file/class at lines 39-113.
- The `slide_windows` method create all windows
- the `slide_windows_and_search_cars` loops through all windows and calculate the HOG features and makes predictions

I have tried several combinations of y start/stop values, windows sizes and overlap and the following combinations lead a sufficient result:
- y start-stop: 400-650
- windows sizes 64, 96 and 128
- overlap of 75%

#### 2. Pipeline

The pipeline exists of the following steps:
- Loop with window size 64x64 through the image and predict per window if car or not
- Same for 96x96 windows and 128x128 windows
- Make a heatmap where for the sum of all previous predictions
- Threshold the heatmap: all values below a certain value (I have used 5 as the threshold) are set to 0 
- Average the heatmap over a the last x number of heatmaps -> I have used 10
- Determine boxes for the final heatmap

Below are some examples of the intermediate steps.

### The output of HOG predictions
![alt text][image4a]
![alt text][image4b]
![alt text][image4c]

### The resulting heatmpa
![alt text][image5]

### The final prediction
![alt text][image6]
---

### Speed improvement

I have tried to improve the speed by calculating the HOG features only ones and aggregating them per window. This did not result in a speedimprovement, so I did not use this.
The code can be found in `find_car_fast.py`

### Video Implementation

#### Final video
Here's a [link to my video result](./test_video_output/project_video.mp4)

#### 2. Remove false positives

I did the following to remove false positives from the video:
- Thresholding the heatmap: remove pixel below a certain count (5)
- Average the heatmap over a number of frames: 10
- Adjusting the window the search for (y between 400 and 650)


