from __future__ import absolute_import
CALIBRATION_PATH = 'output_images/calibrations.p'
import cv2
import pickle
class OutputSaver:
    def __init__(self):
        self.dist_pickle = []
        self.images = {}
    
    def save_calibration(self, mtx, dist):
        self.dist_pickle = {'mtx':mtx, 'dist':dist}

    def save_images(self):
        for path, images in self.images.items():
            for index, image in enumerate(images):
                cv2.imwrite(path % (index+1), image)

    def pickle_calibrations(self):
        pickle.dump( self.dist_pickle, open( CALIBRATION_PATH, 'wb' ) )

    def on_end(self):
        self.pickle_calibrations()
        self.save_images()
    