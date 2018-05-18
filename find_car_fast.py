import numpy as np
import cv2
import time
import os
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

        self.min_car_history = min_car_history
        self.heat_history = []
        self.cnt = 0

    def slide_windows_and_search_cars(self, img, y_start_stop=[None, None], scale=1):
        # img = img.astype(np.float32) / 255

        img_tosearch = img[y_start_stop[0]:y_start_stop[1], :, :]
        mpimg.imsave('{}{}img_tosearch.jpg'.format(self.base_dir, self.count), img_tosearch)
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        starttime = time.time()
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        print(round(time.time() - starttime, 2), 'Seconds to extract HOG features...')

        starttime = time.time()
        found_windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                # features = []
                # features.append(hog_features)

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                # if self.spatial_feat:
                #     spatial_features = bin_spatial(subimg, size=self.spatial_size)
                #     features.append(spatial_features)
                # if self.hist_feat:
                #     hist_features = color_hist(subimg, nbins=self.hist_bins)
                #     features.append(hist_features)

                # Scale features and make a prediction
                # test_features = self.scaler.transform(
                #     np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # features = np.concatenate(features).reshape(1, -1)
                features = hog_features.reshape(1, -1)
                test_features = self.scaler.transform(features)
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    found_windows.append( ((xbox_left, ytop_draw + y_start_stop[0]),
                         (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])))
                    mpimg.imsave('{}car/{}car_{}_{}.jpg'.format(self.base_dir, self.count, xb, yb), subimg)
                else:
                    mpimg.imsave('{}nocar/{}nocar_{}_{}.jpg'.format(self.base_dir, self.count, xb, yb), subimg)
        print(round(time.time() - starttime, 2), 'Seconds to window sliding...')

        return found_windows, []

    def search_cars(self, image, plot=False, write=False, heat_threshold=1, heat_history_max=1,  base_dir='./output_images/'):
        self.count += 1
        self.base_dir = base_dir
        if not os.path.exists(base_dir + 'car'):
            os.makedirs(base_dir + 'car')
        if not os.path.exists(base_dir + 'nocar'):
            os.makedirs(base_dir + 'nocar')
        draw_image = np.copy(image)
        if write:
            mpimg.imsave('{}{}_step0.jpg'.format(base_dir, self.count), draw_image)

        # 64x64
        hot_windows_1, all_windows = self.slide_windows_and_search_cars(image, y_start_stop=[400, 500], scale=1)
        self.window_img_1 = draw_boxes(draw_image, hot_windows_1, color=(0, 255, 0), thick=6)
        if plot:
            plt.imshow(self.window_img_1)
            plt.show()
        if write:
            mpimg.imsave('{}{}_step1.jpg'.format(base_dir, self.count), self.window_img_1)

        # 96x96
        hot_windows_2, all_windows = self.slide_windows_and_search_cars(image, y_start_stop=[400, 600], scale=1.5)
        self.window_img_2 = draw_boxes(draw_image, hot_windows_2, color=(0, 0, 255), thick=6)
        if plot:
            plt.imshow(self.window_img_2)
            plt.show()
        if write:
            mpimg.imsave('{}{}_step2.jpg'.format(base_dir, self.count), self.window_img_2)

        # 128x128
        hot_windows_3, all_windows = self.slide_windows_and_search_cars(image, y_start_stop=[400, 700], scale=2)
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

        if write:
            mpimg.imsave('{}{}-step5_carpos.jpg'.format(base_dir, self.count), draw_img)
            mpimg.imsave('{}{}-step4_heatmap.jpg'.format(base_dir, self.count), self.heatmap_as_normal_img)

        return draw_img

