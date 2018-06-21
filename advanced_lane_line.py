import cv2
import glob
import pickle
import numpy as np

# Camera calibration
# Distortion correction
# Color/gradient threshold
# Perspective transform
NUM_COLOR_CHANNELS = 3
CORNERS_PATH = 'output_images/corners/corners%s.jpg'
ORIGINAL_PATH = 'output_images/original/original%s.jpg'
UNDISTORTED_IMAGES_PATH = 'output_images/undistorted_images/undist%d.jpg'
CALIBRATION_PATH = 'output_images/calibrations.p'
class Camera:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.dist_pickle = []
        self.output_images = {}
        
    def distortion_correction(self, glob_path):
        images = [cv2.imread(path) for path in glob.glob(glob_path)]
        object_points, image_points = self.get_points(images)

        for index, img in enumerate(images):
            undist = self.undistort_image(img, object_points, image_points)
            

    def undistort_image(self, img, objpoints, imgpoints):
        # Function that takes an image, object points, and image points
        # performs the camera calibration, image distortion correction and 
        # returns the undistorted image
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        self.save_calibration(mtx, dist)
        return undist

    def save_calibration(self, mtx, dist):
        self.dist_pickle = {'mtx':mtx, 'dist':dist}

    def save_images(self):
        for path, images in self.output_images.items():
            for index, image in enumerate(images):
                cv2.imwrite(path % str(index+1), image)

    def pickle_calibrations(self):
        pickle.dump( self.dist_pickle, open( CALIBRATION_PATH, 'wb' ) )

    def on_end(self):
        self.pickle_calibrations()
        self.save_images()

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
        self.output_images[CORNERS_PATH] = output_images
        self.output_images[ORIGINAL_PATH] = original_images
        print('Number of Images with corners found: ' + str(len(output_images)))
        return np.array(object_points), np.array(image_points)

    def __get_object_points(self):
        #objp = np.zeros((6*8,3), np.float32)
        #objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
        object_points = np.zeros((self.nx*self.ny, NUM_COLOR_CHANNELS), np.float32)
        object_points[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        return object_points

    def draw_corners(self, index, img, ret, corners): 
        # If found, draw corners
        img = np.copy(img)
        if ret == True:
            cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
            cv2.imwrite(CORNERS_PATH % index, img)
        
            
    def gradient_threshold(self):
        pass
    def perspective_transform(self):
        pass
    
    def convert_to_greyscale(self, img, conversion=cv2.COLOR_BGR2GRAY):
        #Note: Make sure you use the correct grayscale conversion depending on how you've read in your images. 
        # Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
        # Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
        cv2.cvtColor(img, conversion)

if __name__=='__main__':
    camera = Camera(9, 6)
    camera.distortion_correction('camera_cal/calibration*.jpg')
    camera.on_end()
    print('Done')
