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
    #im = cv2.split(im)
    if channels != 3:
        return False
    new_r_im_data, new_g_im_data, new_b_im_data, im_r_arr, im_g_arr, im_b_arr = ([] for i in range(6))
    n_r = 0
    for row in range(height):
        for column in range(width):
            pix_stdev = statistics.stdev(im[row,column])
            if pix_stdev > 8:
                n_r += 1
                b,g,r = im[row,column]
                im_r_arr.append(r)
                im_g_arr.append(g)
                im_b_arr.append(b)
                new_r_im_data.append((r,0,0))
                new_g_im_data.append((0,g,0))
                new_b_im_data.append((0,0,b))
            else:
                im_r_arr.append(0)
                im_g_arr.append(0)
                im_b_arr.append(0)
                new_r_im_data.append((0,0,0))
                new_g_im_data.append((0,0,0))
                new_b_im_data.append((0,0,0))
    
    print "Blue: Skew=", skew(im_b_arr), " Kurtosis=", kurtosis(im_b_arr)
    print "Green Skew: ", skew(im_g_arr), " Kurtosis=", kurtosis(im_g_arr)
    print "Red Skew: ", skew(im_r_arr), " Kurtosis=", kurtosis(im_r_arr)
    ax1 = plt.subplot()
    ax1.hist([im_r_arr,im_g_arr,im_b_arr], bins=np.arange(256))
    ax1.set_ylim(0,1500)
    plt.show()
    #plt.hist(im_g_arr, bins=np.arange(256))
    #plt.hist(im_b_arr, bins=np.arange(256))

    plt.show()

    # for pix in im.getdata():
    #     b,g,r = pix
    #     pix_stdev = statistics.stdev([r,g,b])
    #     if pix_stdev > 8:
    #         n_r += 1
    #         im_r_arr.append(r)
    #         im_g_arr.append(g)
    #         im_b_arr.append(b)
    #         new_r_im_data.append((r,0,0))
    #         new_g_im_data.append((0,g,0))
    #         new_b_im_data.append((0,0,b))
    #     else:
    #         new_r_im_data.append((0,0,0))
    #         new_g_im_data.append((0,0,0))
    #         new_b_im_data.append((0,0,0))

    # img_r = Image.new('RGB', (width, height))
    # img_g = Image.new('RGB', (width, height))
    # img_b = Image.new('RGB', (width, height))
    # img_r.putdata(new_r_im_data)
    # img_g.putdata(new_g_im_data)
    # img_b.putdata(new_b_im_data)

    # print skew(im_r_arr)
    # print skew(im_g_arr)
    # print skew(im_b_arr)
    # # img_r.save('./testing_results/r_'+new_file_name)
    # # img_g.save('./testing_results/g_'+new_file_name)
    # # img_b.save('./testing_results/b_'+new_file_name)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,3,1)
    # ax2 = fig.add_subplot(1,3,2)
    # ax3 = fig.add_subplot(1,3,3)
    # ax1.imshow(img_r)
    # ax2.imshow(img_g)
    # ax3.imshow(img_b)
    # plt.show()

# Description:
# @Param im {PIL.Image Object} - image object converted to grayscale for analysis
def grayProcData(im):
    plt.imshow(im)
    plt.show()

# Description:
# @Param: image{'*.jpg','*.png'} - image to be analyzed and processed into classifier data
# @Param: new_file_name {String} - 

def preProcImage(image,new_file_name):
    test_im = cv2.imread('./testing_images/'+image,1)
    gray_im = cv2.imread('./testing_images/'+image,0)
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