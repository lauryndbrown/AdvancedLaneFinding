from __future__ import absolute_import
import glob
from .gradient_thresholds import GradientThreshold
from .camera_calibrator import CameraCalibrator

# Camera calibration, Distortion correction, Perspective transform
camera = CameraCalibrator(9, 6)
camera.distortion_correction('camera_cal/calibration*.jpg')
camera.on_end()

# Color/gradient threshold
thresh = GradientThreshold()
images = glob.glob('test_images/*.jpg')
for img in images:
    thresh.gradient_threshold(img)

print('Done')
