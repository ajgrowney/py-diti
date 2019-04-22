import cv2
import numpy as np
from scipy.stats import skew, kurtosis

# Description: Official Function In Use by Classifier for Image Processing Data
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


# Description: Developer's function to devise new features to add for classification
def rgbProcDataDevelopment(im):

    results = {}
    height, width, channels = im.shape

    im_b = im.copy()
    im_b[:, :, 1] = 0
    im_b[:, :, 2] = 0

    im_g = im.copy()
    im_g[:, :, 0] = 0
    im_g[:, :, 2] = 0

    im_r = im.copy()
    im_r[:, :, 0] = 0 # b -> 0
    im_r[:, :, 1] = 0 # g -> 0

    # Splitting Image to RGB arrays
    b,g,r = cv2.split(im)
    left_im = im[:,0:(width/2)] # Take left half of the image
    right_im = im[:,(width/2):width] # Take right half of the image
    right_flipped = np.fliplr(right_im) # Flip the right half for image subtraction

    # Image Averages
    image_mean = np.mean(im,axis=(0,1),dtype=np.float64)
    left_mean = np.mean(left_im,axis=(0,1),dtype=np.float64)
    right_mean = np.mean(right_im,axis=(0,1),dtype=np.float64)

    results["right_mean"] = right_mean
    results["left_mean"] = left_mean
    results["image_mean"] = image_mean

    im_sub = np.subtract(right_flipped,left_im)
    im_sub2 = np.subtract(left_im,right_flipped)
    sub_b, sub_g, sub_r = cv2.split(im_sub)


    # Image Subtraction Files
    # cv2.imwrite('./testing_results/im.jpg',im)
    # cv2.imwrite('./testing_results/im_b.jpg',im_b)
    # cv2.imwrite('./testing_results/im_g.jpg',im_g)
    # cv2.imwrite('./testing_results/im_r.jpg',im_r)
    # cv2.imwrite('./testing_results/imsub.jpg',im_sub)
    # cv2.imwrite('./testing_results/imsub2.jpg',im_sub2)
    # cv2.imwrite('./testing_results/imsub_b.jpg',sub_b)
    # cv2.imwrite('./testing_results/imsub_g.jpg',sub_g)
    # cv2.imwrite('./testing_results/imsub_r.jpg',sub_r)

    # # Image Left, Right Segemented Filesskew
    # cv2.imwrite('./testing_results/left.jpg',left_im)
    # cv2.imwrite('./testing_results/right.jpg',right_flipped)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()

    results["b_skew"] = skew(b)
    results["g_skew"] = skew(g)
    results["r_skew"] = skew(r)

    results["b_kurtosis"] = kurtosis(b)
    results["g_kurtosis"] = kurtosis(g)
    results["r_kurtosis"] = kurtosis(r)

    #Plot the histograms below
    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.hist(b, bins=np.arange(256))
    # ax1.set_ylim(0,1500)
    # ax2.hist(g, bins=np.arange(256), color='green')
    # ax2.set_ylim(0,1500)
    # ax3.hist(r, bins=np.arange(256), color='red')
    # ax3.set_ylim(0,1500)
    # plt.show()
    return results
