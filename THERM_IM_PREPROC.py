import sys
import statistics
import json
import csv
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
    current_results["means"]["total_f"]["b"].append(single_im_f['image_mean'][0])
    current_results["means"]["total_f"]["g"].append(single_im_f['image_mean'][1])
    current_results["means"]["total_f"]["r"].append(single_im_f['image_mean'][2])
    current_results["means"]["total_fc"]["b"].append(single_im_fc['image_mean'][0])
    current_results["means"]["total_fc"]["g"].append(single_im_fc['image_mean'][1])
    current_results["means"]["total_fc"]["r"].append(single_im_fc['image_mean'][2])
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
    xf_vals = results["means"]['total_f']["r"]
    yf_vals = results["means"]['total_f']["g"]
    zf_vals = results["means"]['total_f']["b"]

    ax.scatter(xf_vals, yf_vals, zf_vals)
    ax.set_xlabel('R Values')
    ax.set_ylabel('G Values')
    ax.set_zlabel('B Values')

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.set_title("Image Means - FC")
    xfc_vals = results["means"]['total_fc']["r"]
    yfc_vals = results["means"]['total_fc']["g"]
    zfc_vals = results["means"]['total_fc']["b"]

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

    f_imb = cv2.cvtColor(im_b,cv2.COLOR_BGR2GRAY)
    f_img = cv2.cvtColor(im_g,cv2.COLOR_BGR2GRAY)
    f_imr = cv2.cvtColor(im_r,cv2.COLOR_BGR2GRAY)


    f_imb = cv2.medianBlur(f_imb,5)
    f_img = cv2.medianBlur(f_img,5)
    f_imr = cv2.medianBlur(f_imr,5)
    edges = cv2.Canny(f_imr,100,200)

    plt.subplot(2,2,1), plt.imshow(edges)
    plt.subplot(2,2,2), plt.imshow(f_imb)
    plt.subplot(2,2,3), plt.imshow(f_img)
    plt.subplot(2,2,4), plt.imshow(f_imr)

    plt.show()

def writeResultsToCsv(cancer_obj,nocancer_obj, filename):
    f = open(filename,'w')

    with f:
        fieldnames = ['skews_r_f','skews_r_fc','skews_g_f','skews_g_fc', 'skews_b_f','skews_b_fc','kurtosis_r_f','kurtosis_r_fc','kurtosis_g_f','kurtosis_g_fc', 'kurtosis_b_f','kurtosis_b_fc', 'mean_total_f_r', 'mean_total_f_g', 'mean_total_f_b', 'mean_total_fc_r', 'mean_total_fc_g', 'mean_total_fc_b', 'cancer']
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(cancer_obj["means"]["total_f"]["r"])):
            writer.writerow(
                {
                    'skews_r_f': cancer_obj["skews"]["r_f"][i],
                    'skews_r_fc': cancer_obj["skews"]["r_fc"][i],
                    'skews_g_f': cancer_obj["skews"]["g_f"][i],
                    'skews_g_fc': cancer_obj["skews"]["g_fc"][i],
                    'skews_b_f': cancer_obj["skews"]["b_f"][i],
                    'skews_b_fc': cancer_obj["skews"]["b_fc"][i],
                    'kurtosis_r_f': cancer_obj["kurtosis"]["r_f"][i],
                    'kurtosis_r_fc': cancer_obj["kurtosis"]["r_fc"][i],
                    'kurtosis_g_f': cancer_obj["kurtosis"]["g_f"][i],
                    'kurtosis_g_fc': cancer_obj["kurtosis"]["g_fc"][i],
                    'kurtosis_b_f': cancer_obj["kurtosis"]["b_f"][i],
                    'kurtosis_b_fc': cancer_obj["kurtosis"]["b_fc"][i],
                    'mean_total_f_r': cancer_obj["means"]["total_f"]["r"][i],
                    'mean_total_fc_r': cancer_obj["means"]["total_fc"]["r"][i],
                    'mean_total_f_g': cancer_obj["means"]["total_f"]["g"][i],
                    'mean_total_fc_g': cancer_obj["means"]["total_fc"]["g"][i],
                    'mean_total_f_b': cancer_obj["means"]["total_f"]["b"][i],
                    'mean_total_fc_b': cancer_obj["means"]["total_fc"]["b"][i],
                    'cancer': 'Y'
                }
            )

        for j in range(len(nocancer_obj["means"]["total_f"]["r"])):
            writer.writerow(
                {
                    'skews_r_f': nocancer_obj["skews"]["r_f"][j],
                    'skews_r_fc': nocancer_obj["skews"]["r_fc"][j],
                    'skews_g_f': nocancer_obj["skews"]["g_f"][j],
                    'skews_g_fc': nocancer_obj["skews"]["g_fc"][j],
                    'skews_b_f': nocancer_obj["skews"]["b_f"][j],
                    'skews_b_fc': nocancer_obj["skews"]["b_fc"][j],
                    'kurtosis_r_f': nocancer_obj["kurtosis"]["r_f"][j],
                    'kurtosis_r_fc': nocancer_obj["kurtosis"]["r_fc"][j],
                    'kurtosis_g_f': nocancer_obj["kurtosis"]["g_f"][j],
                    'kurtosis_g_fc': nocancer_obj["kurtosis"]["g_fc"][j],
                    'kurtosis_b_f': nocancer_obj["kurtosis"]["b_f"][j],
                    'kurtosis_b_fc': nocancer_obj["kurtosis"]["b_fc"][j],
                    'mean_total_f_r': nocancer_obj["means"]["total_f"]["r"][j],
                    'mean_total_fc_r': nocancer_obj["means"]["total_fc"]["r"][j],
                    'mean_total_f_g': nocancer_obj["means"]["total_f"]["g"][j],
                    'mean_total_fc_g': nocancer_obj["means"]["total_fc"]["g"][j],
                    'mean_total_f_b': nocancer_obj["means"]["total_f"]["b"][j],
                    'mean_total_fc_b': nocancer_obj["means"]["total_fc"]["b"][j],
                    'cancer': 'N'
                }
            )


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
                "total_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "total_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "left_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "left_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "right_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "right_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                }
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
                "total_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "total_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "left_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "left_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "right_f": {
                    "r": [],
                    "g": [],
                    "b": []
                },
                "right_fc": {
                    "r": [],
                    "g": [],
                    "b": []
                }
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
        # displayResults(cancer_results, "Cancer")

        patnocancerlist = open("patients_nocancer.txt", "r").readlines()
        print(len(patnocancerlist))
        for pat in patnocancerlist:
            try:
                pat = pat.strip().replace('\r','')
                pat_res_f,pat_res_fc = preProcImage(pat, "Volunteer_noBG")
                no_cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, no_cancer_results)
            except:
                print("Error")
        # displayResults(no_cancer_results, "No Cancer")
        writeResultsToCsv(cancer_results, no_cancer_results, 'cancer_res_csv2.csv')


    elif arg_len == 2:
        print noiseReduce("EscMar261010","Cancer_NoBG")

if __name__ == '__main__':
    main()
