from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from numpy.polynomial import Polynomial
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product
import shutil
import datetime as dt
from PIL import Image

from matplotlib import pylab

from find_car import FindCar

heat_threshold=3
base_output_dir = './output_images_test_video/ht{}/'.format(heat_threshold)

if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

def process_image(input_img):

   # result = Image.new('RGB', (input_img.shape[1], input_img.shape[0]))
   result = Image.fromarray(find_car.search_cars(input_img, write=True, heat_threshold=heat_threshold, heat_history_max=10, base_dir=base_output_dir))

   window_1 = Image.fromarray(find_car.window_img_1)
   window_1.thumbnail((300, 300))
   result.paste(window_1, (10, 10))

   window_2 = Image.fromarray(find_car.window_img_2)
   window_2.thumbnail((300, 300))
   result.paste(window_2, (320, 10))

   window_3 = Image.fromarray(find_car.window_img_3)
   window_3.thumbnail((300, 300))
   result.paste(window_3, (630, 10))

   heat_window = Image.fromarray(find_car.heatmap_as_normal_img)
   heat_window.thumbnail((300, 300))
   result.paste(heat_window, (940, 10))

   return np.array(result)

videos = [
    # './test_video',
    './project_video',
]



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


for input_file_name in videos:
    time_start = 10
    time_end = 12

    white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)

    clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)
    white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

