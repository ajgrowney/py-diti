import sys
import statistics
import json
import numpy as np
import cv2
from scipy.misc import imread
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA, PCA

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def rgbProcData(im, file_name):

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




# Description:
# @Param im {CV2 image object/ np.ndarray } - image object converted to grayscale for analysis
def grayProcData(im):
    # Edge Detection

    sobel_x = cv2.Sobel(im, cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(im, cv2.CV_64F,0,1,ksize=5)
    laplacian = cv2.Laplacian(im, cv2.CV_64F)
    canny = cv2.Canny(im,300,200)
    plt.subplot(2,2,1), plt.imshow(im,cmap='gray')
    plt.subplot(2,2,2), plt.imshow(sobel_y, cmap='gray')
    plt.subplot(2,2,3), plt.imImagesshow(laplacian)
    plt.subplot(2,2,4), plt.imshow(canny)
    plt.show()
    return

# Description:
# @Param: image{'*.jpg','*.png'} - image to be analyzed and processed into classifier data
# @Param: new_file_name {String} -

def preProcImage(image, folder):
    f_im = cv2.imread('./'+folder+'/'+image+'A2BA-f.jpg',1)
    fc_im = cv2.imread('./'+folder+'/'+image+'A2BA-fc.jpg',1)


    # Utilize the RGB Colorspace to analyze the image and pull statistics
    rgb_proc_f = rgbProcData(f_im, image)
    rgb_proc_fc = rgbProcData(fc_im,image)

    # # fname = './testing_results/json_results/'+image+'_f_data.txt'
    # # fcname = './testing_results/json_results/'+image+'_fc_data.txt'
    # # json.dump(rgb_proc_f, open(fname, 'w'), cls=NumpyEncoder)
    # # json.dump(rgb_proc_fc, open(fcname, 'w'), cls=NumpyEncoder)

    return rgb_proc_f, rgb_proc_fc

def compileHistogramResults(single_im_f, single_im_fc, current_results):
    current_results["skews"]["r_fc"].append(single_im_fc['r_skew'])
    current_results["skews"]["r_f"].append(single_im_f['r_skew'])
    current_results["skews"]["b_fc"].append(single_im_fc['b_skew'])
    current_results["skews"]["b_f"].append(single_im_f['b_skew'])
    current_results["skews"]["g_fc"].append(single_im_fc['g_skew'])
    current_results["skews"]["g_f"].append(single_im_f['g_skew'])
    current_results["kurtosis"]["r_fc"].append(single_im_fc['r_kurtosis'])
    current_results["kurtosis"]["r_f"].append(single_im_f['r_kurtosis'])
    current_results["kurtosis"]["b_fc"].append(single_im_fc['b_kurtosis'])
    current_results["kurtosis"]["b_f"].append(single_im_f['b_kurtosis'])
    current_results["kurtosis"]["g_fc"].append(single_im_fc['g_kurtosis'])
    current_results["kurtosis"]["g_f"].append(single_im_f['g_kurtosis'])
    current_results["means"]["total_f"].append(single_im_f['image_mean'])
    current_results["means"]["total_fc"].append(single_im_fc['image_mean'])
    return current_results

def displayResults(results, title):
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.set_title(title + " Skew Data")
    skew_data = [results["skews"]["r_f"], results["skews"]["r_fc"], results["skews"]["g_f"], results["skews"]["g_fc"], results["skews"]["b_f"], results["skews"]["b_fc"]]
    colors = ['red', 'crimson','green', 'yellow','blue', 'skyblue']
    n, bins, patches = plt.hist(skew_data, 10, color=colors)

    ax = fig.add_subplot(2,2,2)
    ax.set_title(title + " Kurtosis Data")
    kurt_data = [results["kurtosis"]["r_f"], results["kurtosis"]["r_fc"], results["kurtosis"]["g_f"], results["kurtosis"]["g_fc"], results["kurtosis"]["b_f"], results["kurtosis"]["b_fc"]]
    colors = ['red', 'crimson','green', 'yellow','blue', 'skyblue']
    n, bins, patches = plt.hist(kurt_data, 10, color=colors)

    ax = fig.add_subplot(2,2,3, projection='3d')
    ax.set_title("Image Means - F")
    xf_vals = list(x[0] for x in results["means"]['total_f'])
    yf_vals = list(x[1] for x in results["means"]['total_f'])
    zf_vals = list(x[2] for x in results["means"]['total_f'])

    ax.scatter(xf_vals, yf_vals, zf_vals)
    ax.set_xlabel('R Values')
    ax.set_ylabel('G Values')
    ax.set_zlabel('B Values')

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.set_title("Image Means - FC")
    xfc_vals = list(x[0] for x in results["means"]['total_fc'])
    yfc_vals = list(x[1] for x in results["means"]['total_fc'])
    zfc_vals = list(x[2] for x in results["means"]['total_fc'])

    ax.scatter(xfc_vals, yfc_vals, zfc_vals)
    ax.set_xlabel('R Values')
    ax.set_ylabel('G Values')
    ax.set_zlabel('B Values')
    plt.show()


def noiseReduce(image, folder):
    f_im = cv2.imread('./'+folder+'/'+image+'A2BA-f.jpg',1)
    fc_im = cv2.imread('./'+folder+'/'+image+'A2BA-fc.jpg',0)

    im_b = f_im.copy()
    im_b[:, :, 1] = 0
    im_b[:, :, 2] = 0

    im_g = f_im.copy()
    im_g[:, :, 0] = 0
    im_g[:, :, 2] = 0

    im_r = f_im.copy()
    im_r[:, :, 0] = 0 # b -> 0
    im_r[:, :, 1] = 0 # g -> 0

    f_im = cv2.cvtColor(im_b,cv2.COLOR_BGR2GRAY)

    f_im2 = cv2.medianBlur(f_im,5)
    edges = cv2.Canny(f_im2,100,200)

    plt.subplot(1,2,1), plt.imshow(f_im2)
    plt.subplot(1,2,2), plt.imshow(edges)
    plt.show()

def main():
    arg_len = len(sys.argv)

    if arg_len == 2 and sys.argv[1] == "frontDataAll":
        # Results Object
        cancer_results = {
            "skews": {
                "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
            },
            "kurtosis": {
                "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
            },
            "means": {
                "total_f": [], "total_fc": [], "left_f": [], "left_fc": [], "right_f": [], "right_fc": []
            }
        }
        no_cancer_results = {
            "skews": {
                "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
            },
            "kurtosis": {
                "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
            },
            "means": {
                "total_f": [], "total_fc": [], "left_f": [], "left_fc": [], "right_f": [], "right_fc": []
            }
        }

        patlist = open("patients_cancer.txt", "r").readlines()
        for pat in patlist:
            try:
                pat = pat.strip().replace('\r','')
                pat = pat[1::2]
                pat = pat[:-1]
                pat_res_f, pat_res_fc = preProcImage(pat, "Cancer_noBG")
                cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, cancer_results)
            except:
                print("Error", pat.strip().replace('\r',''))
        displayResults(cancer_results, "Cancer")

        patnocancerlist = open("patients_nocancer.txt", "r").readlines()
        print(len(patnocancerlist))
        for pat in patnocancerlist:
            try:
                pat = pat.strip().replace('\r','')
                pat_res_f,pat_res_fc = preProcImage(pat, "Volunteer_noBG")
                no_cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, no_cancer_results)
            except:
                print("Error")
        displayResults(no_cancer_results, "No Cancer")


    elif arg_len == 2:
        print noiseReduce("AcoAlm221112","Volunteer_noBG")

if __name__ == '__main__':
    main()
