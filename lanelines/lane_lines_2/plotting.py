
import matplotlib.pyplot as plt
import numpy as np

def draw_lines():
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)


def show_image(image):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    ax2.imshow(hls)
    ax2.set_title('HLS', fontsize=50)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    ax3.imshow(hsv)
    ax3.set_title('HSV', fontsize=50)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def plot_side_by_side(image1, image2, label1='Original Image', label2='Thresholded Grad. Dir.'):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(label1, fontsize=50)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title(label2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def plot_hist(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    plt.plot(histogram)
    plt.show()
