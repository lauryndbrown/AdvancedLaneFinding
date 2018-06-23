#from __future__ import absolute_import
import cv2
import glob

import numpy as np

from output_saver import OutputSaver

NUM_COLOR_CHANNELS = 3
CORNERS_PATH = '../output_images/corners/corners%d.jpg'
ORIGINAL_PATH = '../output_images/original/original%d.jpg'
UNDISTORTED_PATH = '../output_images/undistorted/undistorted%d.jpg'
WARPED_PATH = '../output_images/warped/warped%d.jpg'

class CameraCalibrator:
    def __init__(self, nx, ny, output_saver=None):
        self.nx = nx
        self.ny = ny
        self.output_saver = OutputSaver() if output_saver is None else output_saver
        
        
    def distortion_correction(self, glob_path):
        images = [cv2.imread(path) for path in glob.glob(glob_path)]
        object_points, image_points = self.get_points(images)
        undist_images = []
        warped_images = []

        for index, img in enumerate(self.output_saver.images[ORIGINAL_PATH]):
            undist = self.undistort_image(img, object_points, image_points)
            undist_images.append(undist)
            warped, M = self.perspective_transform(undist, image_points[index])
            warped_images.append(warped)
        self.output_saver.images[UNDISTORTED_PATH] = undist_images
        self.output_saver.images[WARPED_PATH] = warped_images
            
    def undistort_image(self, img, objpoints, imgpoints):
        # Function that takes an image, object points, and image points
        # performs the camera calibration, image distortion correction and 
        # returns the undistorted image
        #img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        self.output_saver.add_calibration(mtx, dist)
        return undist

    

    def get_points(self, images):
        object_points = [] 
        image_points = []
        original_images = []
        output_images = []
        objp = self.__get_object_points()
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret is True:
                img_corners = cv2.drawChessboardCorners(img.copy(), (self.nx, self.ny), corners, ret)
                image_points.append(corners)
                object_points.append(objp)
                output_images.append(img_corners)
                original_images.append(img)
        self.output_saver.images[CORNERS_PATH] = output_images
        self.output_saver.images[ORIGINAL_PATH] = original_images
        print('Number of Images with corners found: ' + str(len(output_images)))
        return np.array(object_points), np.array(image_points)

    def __get_object_points(self):
        #objp = np.zeros((6*8,3), np.float32)
        #objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
        object_points = np.zeros((self.nx*self.ny, NUM_COLOR_CHANNELS), np.float32)
        object_points[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        return object_points

    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients
    def perspective_transform(self, undist, corners):
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[self.nx-1], corners[-1], corners[-self.nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                    [img_size[0]-offset, img_size[1]-offset], 
                                    [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

        # Return the resulting image and matrix
        return warped, M