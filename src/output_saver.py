from __future__ import absolute_import

import cv2
import matplotlib.pyplot as plt
import pickle


class OutputSaver:
    def __init__(self):
        self.dist_pickle = []
        self.images = {}
    
    def add_calibration(self, mtx, dist):
        self.dist_pickle = {'mtx':mtx, 'dist':dist}

    def save_images(self):
        for path, images in self.images.items():
            for index, image in enumerate(images):
                cv2.imwrite(path % (index+1), image)

    def pickle_calibrations(self, path):
        pickle.dump( self.dist_pickle, open( path, 'wb' ) )

    def plot_images(self, before_name, before_image, after_name, after_image):
        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title(before_name)
        ax1.imshow(before_image)

        ax2.set_title(after_name)
        ax2.imshow(after_image, cmap='gray')

    def on_end(self, calibration_path):
        self.pickle_calibrations(calibration_path)
        self.save_images()
    