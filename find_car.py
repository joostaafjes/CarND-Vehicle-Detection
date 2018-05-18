import numpy as np
import cv2
import time
from skimage.feature import hog
from functions import *
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

class FindCar:

    def __init__(self, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True, min_car_history=1):
        self.clf = clf
        self.scaler = scaler
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

        self.count = 0

        self.heat_history = []
        self.min_car_history = min_car_history
        self.cnt = 0

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_windows(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

        # Return the list of windows
        self.window_list = window_list
        return window_list

    def slide_windows_and_search_cars(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                         xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        self.slide_windows(img, x_start_stop, y_start_stop, xy_window, xy_overlap)
        return self.search_windows(img)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        starttime = time.time()
        for window in self.window_list:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img, color_space=self.color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            # 5) Scale extracted features to be fed to classifier
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = self.clf.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
                # mpimg.imsave('{}{:04d}_extra.jpeg'.format('./stills/', self.cnt), test_img)
                # self.cnt+=1
            # else:
            #     mpimg.imsave('{}{:04d}_car.jpeg'.format('./stills/', self.cnt), test_img)
            #     self.cnt+=1
        print('- ', round(time.time() - starttime, 2), 'Seconds to extract HOG features...')
        # 8) Return windows for positive detections
        return on_windows, self.window_list

    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)
        # 3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(self, feature_image, size=spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(self, feature_image, nbins=hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)

    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)



    def search_cars(self, image, plot=False, write=False, heat_threshold=1, heat_history_max=1,  base_dir='./output_images/'):
        self.count += 1
        draw_image = np.copy(image)
        if write:
            mpimg.imsave('{}{}_step0.jpg'.format(base_dir, self.count), draw_image)

        # 64x64
        hot_windows_1, all_windows = self.slide_windows_and_search_cars(image,
                                                                            x_start_stop=[None, None],
                                                                            y_start_stop=[400, 650],
                                                                            xy_window=(64, 64), xy_overlap=(0.75, 0.75))
        self.window_img_1 = draw_boxes(draw_image, hot_windows_1, color=(0, 255, 0), thick=6)
        if plot:
            plt.imshow(self.window_img_1)
            plt.show()
        if write:
            mpimg.imsave('{}{}_step1.jpg'.format(base_dir, self.count), self.window_img_1)

        # 96x96
        hot_windows_2, all_windows = self.slide_windows_and_search_cars(image,
                                                                            x_start_stop=[None, None],
                                                                            y_start_stop=[400, 650],
                                                                            xy_window=(96, 96), xy_overlap=(0.75, 0.75))
        self.window_img_2 = draw_boxes(draw_image, hot_windows_2, color=(0, 0, 255), thick=6)
        if plot:
            plt.imshow(self.window_img_2)
            plt.show()
        if write:
            mpimg.imsave('{}{}_step2.jpg'.format(base_dir, self.count), self.window_img_2)

        # 128x128
        hot_windows_3, all_windows = self.slide_windows_and_search_cars(image,
                                                                            x_start_stop=[None, None],
                                                                            y_start_stop=[400, 650],
                                                                            xy_window=(128, 128), xy_overlap=(0.75, 0.75))
        self.window_img_3 = draw_boxes(draw_image, hot_windows_3, color=(255, 0, 0), thick=6)
        if plot:
            plt.imshow(self.window_img_3)
            plt.show()
        if write:
            mpimg.imsave('{}{}_step3.jpg'.format(base_dir, self.count), self.window_img_3)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each window in hot_windows
        heat = add_heat(heat, hot_windows_1)
        heat = add_heat(heat, hot_windows_2)
        heat = add_heat(heat, hot_windows_3)

        # update history
        if len(self.heat_history) >= heat_history_max:
            self.heat_history.pop(0)
        self.heat_history.append(heat)
        heat = np.average(self.heat_history, axis=0)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, heat_threshold)

        # Visualize the heatmap when displaying
        self.heatmap = np.clip(heat, 0, 255)
        self.heatmap_as_normal_img = 255 * self.heatmap / self.heatmap.max()

        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        self.cars_found = labels[1]

        # wait till some history is available
        if len(self.heat_history) < self.min_car_history:
            draw_img = np.copy(image)
        else:
            draw_img = draw_labeled_bboxes(np.copy(image), labels)

        mpimg.imsave('{}{}-step5_carpos.jpg'.format(base_dir, self.count), draw_img)
        mpimg.imsave('{}{}-step4_heatmap.jpg'.format(base_dir, self.count), self.heatmap_as_normal_img)

        return draw_img

