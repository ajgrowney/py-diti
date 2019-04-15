import numpy as np
import cv2
from scipy.stats import skew, kurtosis

# Param: im { Numpy Array } - contains the image read
# Return: results { List } - check documentation for details on each element in the list
def rgbProcData(im):
    height, width, channels = im.shape
    b,g,r = cv2.split(im)
    left_im = im[:,0:(width/2)]
    right_im = im[:,(width/2):width]
    right_flipped = np.fliplr(right_im)

    image_mean = np.mean(im,axis=(0,1),dtype=np.float64)
    left_mean = np.mean(left_im,axis=(0,1),dtype=np.float64)
    right_mean = np.mean(right_im,axis=(0,1),dtype=np.float64)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()

    b_skew, g_skew, r_skew = skew(b), skew(g), skew(r)
    b_kurt, g_kurt, r_kurt = kurtosis(b), kurtosis(g), kurtosis(r)

    results = [r_skew, g_skew, b_skew, r_kurt, g_kurt, b_kurt, image_mean[0], image_mean[1], image_mean[2], left_mean[0], left_mean[1], left_mean[2], right_mean[0], right_mean[1], right_mean[2]]
    return np.array(results)
