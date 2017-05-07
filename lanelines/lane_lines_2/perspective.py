


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob


class Perspective(object):
    def __init__(self, image=None):
        self.M = None
        self.Minv = None
        self.image = None

    def fit(self, image=None):
        # Grab the image shape

        if image is not None:
            self.image = image
        img_shape = (image.shape[1], image.shape[0])

        height = img_shape[1]
        width = img_shape[0]

        offset = 0
        # top_center_dist = .03
        # bottom_center_dist = .32
        # top_center_dist = .05
        # bottom_center_dist = .34
        #
        top_center_dist = .04
        bottom_center_dist = .35

        src = np.float32([[width * (0.5  -  top_center_dist )+ offset,height * 0.62],
                        [width * ( 0.5 - bottom_center_dist) +offset, height * 0.95],
                        [width * (0.5  + bottom_center_dist)+ offset, height * 0.95],
                        [width * (0.5  + top_center_dist) + offset, height * 0.62]])
        dst = np.float32([[width * ( 0.5 - bottom_center_dist) + offset, height * 0.0],
                        [width * ( 0.5 - bottom_center_dist) + offset, height * 0.95],
                        [width * (0.5  + bottom_center_dist) + offset, height * 0.95],
                        [width * (0.5  + bottom_center_dist) + offset, height * 0.0]])

        pts = src.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(image,[pts],True,(0,255,255))

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        self.M = M
        self.Minv = Minv

    def warp(self):

        image = self.image
        # offset = 10 # offset for dst points
        # Grab the image shape
        img_shape = (image.shape[1], image.shape[0])

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(image, self.M, img_shape, flags=cv2.INTER_LINEAR)
        return warped


    def unwarp(self):
        image = self.image
        # offset = 10 # offset for dst points
        # Grab the image shape
        img_shape = (image.shape[1], image.shape[0])

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        unwarped = cv2.warpPerspective(image, self.Minv, img_shape, flags=cv2.INTER_LINEAR)
        return unwarped

def get_perspective_transform(image):
    # offset = 10 # offset for dst points
    # Grab the image shape
    img_shape = (image.shape[1], image.shape[0])

    height = img_shape[1]
    width = img_shape[0]

    offset = 0
    # top_center_dist = .03
    # bottom_center_dist = .32
    # top_center_dist = .05
    # bottom_center_dist = .34
    #
    top_center_dist = .04
    bottom_center_dist = .35

    src = np.float32([[width * (0.5  -  top_center_dist )+ offset,height * 0.62],
                    [width * ( 0.5 - bottom_center_dist) +offset, height * 0.95],
                    [width * (0.5  + bottom_center_dist)+ offset, height * 0.95],
                    [width * (0.5  + top_center_dist) + offset, height * 0.62]])
    dst = np.float32([[width * ( 0.5 - bottom_center_dist) + offset, height * 0.0],
                    [width * ( 0.5 - bottom_center_dist) + offset, height * 0.95],
                    [width * (0.5  + bottom_center_dist) + offset, height * 0.95],
                    [width * (0.5  + bottom_center_dist) + offset, height * 0.0]])

    pts = src.astype(np.int32).reshape((-1,1,2))
    cv2.polylines(image,[pts],True,(0,255,255))

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(image, M, img_shape, flags=cv2.INTER_LINEAR)
    return warped, Minv
