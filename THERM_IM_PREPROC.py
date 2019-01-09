import sys
import statistics
import heapq
import numpy as np
import cv2
from scipy.misc import imread
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import FastICA, PCA


def rgbProcData(im, new_file_name):
    height, width, channels = im.shape
    print np.mean(im[240:480,0:20],axis=(0,1),dtype=np.float64)
    low = np.array([120,0,120])
    up = np.array([255,15,255])
    mask = cv2.inRange(im,low,up)
    out = cv2.bitwise_and(im,im,mask=mask) # Storing the initial mask

    print((out.shape))

    im_bw = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

    im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    big_four = heapq.nlargest(4,contour_sizes,key=lambda x: x[0])
    big_four_contours = list(x[1] for x in big_four)

    print len(big_four)

    cv2.drawContours(out, big_four_contours, -1, (0,255,0), -1)

    for x in xrange(0,height):
        for y in xrange(0,width):
            if (out[x][y] == [0,255,0]).all():
                im[x][y] = [0,0,0]


    b,g,r = cv2.split(im)
    left_im = im[:,0:(width/2)] # Take left half of the image
    right_im = im[:,(width/2):width] # Take right half of the image
    right_flipped = np.fliplr(right_im) # Flip the right half for image subtraction

    # Image Averages
    print("Image Mean:",np.mean(im,axis=(0,1),dtype=np.float64))
    print("Left Image:",np.mean(left_im,axis=(0,1),dtype=np.float64))
    print("Right Image: ",np.mean(right_im,axis=(0,1),dtype=np.float64))

    im_sub = np.subtract(right_flipped,left_im)
    im_sub2 = np.subtract(left_im,right_flipped)
    sub_b, sub_g, sub_r = cv2.split(im_sub)


    # Image Subtraction Files
    cv2.imwrite('./testing_results/im_out.jpg',out)
    #cv2.imwrite('./testing_results/im_maskblur.jpg',im_blur)
    cv2.imwrite('./testing_results/im_updated.jpg',im)
    cv2.imwrite('./testing_results/imgray.jpg',im_bw)
    cv2.imwrite('./testing_results/imsub.jpg',im_sub)
    cv2.imwrite('./testing_results/imsub2.jpg',im_sub2)
    cv2.imwrite('./testing_results/imsub_b.jpg',sub_b)
    cv2.imwrite('./testing_results/imsub_g.jpg',sub_g)
    cv2.imwrite('./testing_results/imsub_r.jpg',sub_r)

    # Image Left, Right Segemented Files
    cv2.imwrite('./testing_results/left.jpg',left_im)
    cv2.imwrite('./testing_results/right.jpg',right_flipped)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    print "Blue: Skew=", skew(b), " Kurtosis=", kurtosis(b)
    print "Green Skew: ", skew(g), " Kurtosis=", kurtosis(g)
    print "Red Skew: ", skew(r), " Kurtosis=", kurtosis(r)
    # Plot the histograms below
    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.hist(b, bins=np.arange(256))
    # ax1.set_ylim(0,1500)
    # ax2.hist(g, bins=np.arange(256), color='green')
    # ax2.set_ylim(0,1500)
    # ax3.hist(r, bins=np.arange(256), color='red')
    # ax3.set_ylim(0,1500)
    # plt.show()
    return




# Description:
# @Param im {CV2 image object/ np.ndarray } - image object converted to grayscale for analysis
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
    test_im = cv2.imread('./testing_images/'+image,1)
    gray_im = cv2.imread('./testing_images/'+image,cv2.IMREAD_GRAYSCALE)
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
