import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label

from find_car import FindCar

# load a pe-trained svc model from a serialized (pickle) file
dist_pickle = pickle.load(open("svc_pickle.p", "rb"))

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"] # 9
pix_per_cell = dist_pickle["pix_per_cell"] # 8
cell_per_block = dist_pickle["cell_per_block"] # 2
spatial_size = dist_pickle["spatial_size"] # 32x32
hist_bins = dist_pickle["hist_bins"] # 32

# img = mpimg.imread('test_image.jpg')
image = mpimg.imread('bbox-example-image.jpg')
draw_image = np.copy(image)

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True  # HOG features on or off

find_car = FindCar(svc, X_scaler, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

hot_windows, all_windows = find_car.slide_windows_and_search_cars(image,
                                               x_start_stop=[None, None], y_start_stop=[400, 720],
                                               xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)

plt.imshow(window_img)
plt.show()

heat = np.zeros_like(image[:, :, 0]).astype(np.float)

# Add heat to each window in hot_windows
heat = add_heat(heat, hot_windows)

# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)

# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
cars_found = labels[1]
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions ({})'.format(cars_found))
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()

