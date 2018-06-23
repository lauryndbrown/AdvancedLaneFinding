from __future__ import absolute_import
import cv2
import glob
import pickle
import numpy as np


MAX_COLOR_VALUE = 255


class GradientThreshold:
    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def apply_threshold(self, thresh_min, thresh_max, channel):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        return binary

    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=MAX_COLOR_VALUE):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = self.apply_threshold(thresh_min=0, thresh_max=MAX_COLOR_VALUE, channel=scaled_sobel)
        return binary_output
    
    def combine_binary(self, binary1, binary2):
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(binary2)
        combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
        return combined_binary
    
    # Define a function that thresholds the S-channel of HLS
    def hls_select(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def gradient_threshold(self, img):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        sxbinary = self.abs_sobel_thresh(img, thresh_min=20, thresh_max=100)
        s_binary = self.apply_threshold(thresh_min=170, thresh_max=255, channel=s_channel)

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * MAX_COLOR_VALUE
        combined_binary = self.combine_binary(s_binary, sxbinary)

        #self.plot_threshold_images(color_binary, combined_binary)
        return color_binary, combined_binary
        
        