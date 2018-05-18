import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
import time
import ntpath
from functions import *

# from find_car import FindCar
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

# image_list = glob.glob('./test_images/new6/*.jpg', recursive=True)
image_list = [
#     'bbox-example-image',
#     'test_image',
#     'test1',
#     'test2',
#     'test3',
#     'test4',
#     'test5',
#     'test6',
#     '1_step0',
#     '2_step0',
#     '3_step0',
#     '495_step0',
#     '4_step0',
#     '620_step0',
#     '630_step0',
#     'new/499_step0',
#     'new/506_step0',
#     'new/615_step0',
#     'new/640_step0',
#     'new/656_step0',
#     'new/677_step0',
#     'new/688_step0',
#     'new2/373_step0',
#     'new2/377_step0',
#     'new2/380_step0',
#     'new2/493_step0',
#     'new2/497_step0',
#     'new2/502_step0',
#     'new3/619_step0',
#     './test_images/new3/622_step0.jpg',
#     'new3/626_step0',
#     'new3/634_step0',
#     'new3/654_step0',
#     './test_images/new3/656_step0.jpg',
#     './test_images/new7/1160_step0.jpg',
#     './test_images/new7/1170_step0.jpg',
    './test_images/new7/1180_step0.jpg',
]

for image_filename in image_list:
    for heat_threshold in range(1, 2):
        starttime = time.time()
        # img = mpimg.imread('test_image.jpg')
        base_output_dir = './output_images2/ht{}/'.format(heat_threshold)
        image_base_filename = ntpath.basename(image_filename)
        image = mpimg.imread(image_filename)

        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        draw_img = find_car.search_cars(image, plot=True, write=True, heat_threshold=heat_threshold, heat_history_max=1, base_dir=base_output_dir)
        print(round(time.time() - starttime, 2), 'Seconds after find car...')

        # 0.2 sec
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions ({})'.format(find_car.cars_found))
        plt.subplot(122)
        plt.imshow(find_car.heatmap, cmap='hot')
        plt.title('Heat Map ' + image_base_filename)
        fig.tight_layout()
        plt.show()

        print(round(time.time() - starttime, 2), 'Seconds overall...')

