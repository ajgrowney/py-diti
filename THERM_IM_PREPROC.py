import sys
import statistics
import numpy as np
import cv2
from scipy.misc import imread
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import FastICA, PCA


def rgbProcData(im, new_file_name):
    height, width, channels = im.shape
    print height, width, channels
    b,g,r = cv2.split(im)
    colors = ("b","g","r")
    if channels != 3:
        return False


    new_r_im_data, new_g_im_data, new_b_im_data, im_r_arr, im_g_arr, im_b_arr = ([] for i in range(6))
    left_im = { "red": [], "blue": [], "green": [], "total": []}
    right_im = { "red": [], "blue": [], "green": [], "total": []}

    # mid: segment the image into left and right side
    mid = width/2
    valid_pixs = []


    # Traverse the rows of the image
    for row in range(height):
        inner_pixs = []

        # Traverse each column in that row of the image
        # TODO: Add left and right segmentation
        for column in range(width):
            pix_stdev = statistics.stdev(im[row,column])
            if pix_stdev > 8:
                inner_pixs.append(column)
                b,g,r = im[row,column]
                im_r_arr.append(r)
                im_g_arr.append(g)
                im_b_arr.append(b)
                new_r_im_data.append((r,0,0))
                new_g_im_data.append((0,g,0))
                new_b_im_data.append((0,0,b))

                if column < mid:
                    left_im["total"].append(im[row,column])
                else:
                    right_im["total"].append(im[row,column])
            else:
                im_r_arr.append(0)
                im_g_arr.append(0)
                im_b_arr.append(0)
                new_r_im_data.append((0,0,0))
                new_g_im_data.append((0,0,0))
                new_b_im_data.append((0,0,0))

                if column < mid:
                    left_im["total"].append(im[row,column])
                else:
                    right_im["total"].append(im[row,column])
        valid_pixs.append(inner_pixs)

    # Total Image Statistics
    print "Blue: Skew=", skew(im_b_arr), " Kurtosis=", kurtosis(im_b_arr)
    print "Green: Skew= ", skew(im_g_arr), " Kurtosis=", kurtosis(im_g_arr)
    print "Red: Skew= ", skew(im_r_arr), " Kurtosis=", kurtosis(im_r_arr)

    # Left Side Statistics
    total_sum_tuple = reduce(lambda sum_tuple, x: tuple(map((lambda y,z: y+z),x,sum_tuple)), left_im["total"])
    print total_sum_tuple
    print "Num Valid Pix", len(valid_pixs)

#    ax1 = plt.subplot()
#    ax1.hist([im_r_arr,im_g_arr,im_b_arr], bins=np.arange(256))
#    ax1.set_ylim(0,1500)
#    plt.show()


    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    print "Blue: Skew=", skew(b), " Kurtosis=", kurtosis(b)
    print "Green Skew: ", skew(g), " Kurtosis=", kurtosis(g)
    print "Red Skew: ", skew(r), " Kurtosis=", kurtosis(r)


# Description:
# @Param im {PIL.Image Object} - image object converted to grayscale for analysis
def grayProcData(im):
        # Edge Detection

    sobel_x = cv2.Sobel(im, cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(im, cv2.CV_64F,0,1,ksize=5)
    laplacian = cv2.Laplacian(im, cv2.CV_64F)
    canny = cv2.Canny(im,300,200)
    #plt.subplot(2,2,1), plt.imshow(sobel_x,cmap='gray')
    #plt.subplot(2,2,2), plt.imshow(sobel_y, cmap='gray')
    #plt.subplot(2,2,3), plt.imshow(laplacian)
    #plt.subplot(2,2,4), plt.imshow(canny)
    #plt.show()
    return

# Description:
# @Param: image{'*.jpg','*.png'} - image to be analyzed and processed into classifier data
# @Param: new_file_name {String} -

def preProcImage(image,new_file_name):
    test_im = cv2.imread('../Images/'+image,1)
    gray_im = cv2.imread('./Images/'+image,cv2.IMREAD_GRAYSCALE)
    #im_ycbcr = imread(image,mode='YCbCr')

    # Utilize the RGB Colorspace to analyze the image and pull statistics
    rgb_proc = rgbProcData(test_im, new_file_name)
    if(rgb_proc == False):
        print "Error in RGB Processing"

    grayProcData(gray_im)

    # Independent Component Analysis
    # transformer = FastICA(n_components=3)
    # im_ica = transformer.fit(new_r_im_data)
    # im_restored = transformer.inverse_transform(im_ica)


def main():
    arg_len = len(sys.argv)

    if arg_len == 3:
        preProcImage(sys.argv[1],sys.argv[2])
        print "We here"

if __name__ == '__main__':
    main()
