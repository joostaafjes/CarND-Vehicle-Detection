
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
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

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
- y start-stop: 400-720
- windows sizes 64, 96 and 128
- overlap of 75%

#### 2. Pipeline

- heatmap history
- threshold


![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

