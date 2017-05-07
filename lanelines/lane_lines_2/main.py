
import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt

import matplotlib.image as mpimg


def main():
    # image = mpimg.imread('../img/signs_vehicles_xygrad.jpg')
    image = mpimg.imread('./test_images/straight_lines1.jpg')
    # image = mpimg.imread('../test_images/test1.jpg')
    plt.imshow(image)

    # calibrate

    cal_mages = glob.glob('./camera_cal/calibration*.jpg')
    objpoints, imgpoints = calibration.calibrate(images=cal_mages)

    undistorted = calibration.cal_undistort(image, objpoints, imgpoints)
    plotting.plot_side_by_side(image, undistorted, label1='Original Image', label2='undistorted')

    # apply thresholds
    thresh_img = thresholds.apply_gradient_thresholds(image, ksize=5, sobel_thresh=(90,250))
    plotting.plot_side_by_side(undistorted, thresh_img, label1='Undistorted', label2='thresholded')

    # perspective transform
    warped = perspective.get_perspective_transform(thresh_img)
    plotting.plot_side_by_side(undistorted, warped, label1='Undistorted', label2='perspective transform')
