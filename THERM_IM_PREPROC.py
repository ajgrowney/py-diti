import sys
import statistics
import json
import csv
import numpy as np
import pandas as pd
import cv2
from scipy.misc import imread
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA, PCA

from Algorithms.transformations import noiseReduce, retinex
from Algorithms.imageCSVConversion import csvToImage, imageToCSV
from Algorithms.rgbData import rgbProcDataDevelopment



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
    b,g,r = cv2.split(f_im)
    f_im = cv2.merge([r,g,b])
    fc_im = cv2.imread('./'+folder+'/'+image+'A2BA-fc.jpg',1)
    b,g,r = cv2.split(fc_im)
    fc_im = cv2.merge([r,g,b])


    # Utilize the RGB Colorspace to analyze the image and pull statistics
    rgb_proc_f = rgbProcDataDevelopment(f_im)
    rgb_proc_fc = rgbProcDataDevelopment(fc_im)

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



def singleImHist(im, display=False, limit=None):
    b,g,r = cv2.split(im)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    if display:
        # Plot the histograms below
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        fig.suptitle("Single Image Histogram")
        ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
        ax2.hist(g, bins=np.arange(256), color='green', ec="green")
        ax3.hist(r, bins=np.arange(256), color="red",ec="red")

        if limit != None:
            ax1.set_ylim(0,limit)
            ax2.set_ylim(0,limit)
            ax3.set_ylim(0,limit)
        plt.show()

    return b,g,r

def totalImHist(im,b_total,g_total,r_total):
    b,g,r = cv2.split(im)
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    b_total = np.concatenate([b_total,b])
    g_total = np.concatenate([g_total,g])
    r_total = np.concatenate([r_total,r])
    return b_total, g_total, r_total

def showtotalImHist(b,g,r,limit=None):
    # Plot the histograms below
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.suptitle("Total Image Histogram")
    ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
    ax2.hist(g, bins=np.arange(256), color='green', ec="green")
    ax3.hist(r, bins=np.arange(256), color="red",ec="red")

    if limit != None:
        ax1.set_ylim(0,limit)
        ax2.set_ylim(0,limit)
        ax3.set_ylim(0,limit)

    plt.show()


def getPatients(patients_path):
    patlist = open(patients_path, "r").readlines()
    patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
    return patlist

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

        # Get patient id lists to retrieve image files
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        for pat in patlist:
            try:
                pat_res_f, pat_res_fc = preProcImage(pat, "Images_noBG")
                cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, cancer_results)
            except:
                print("Error", pat.strip().replace('\r',''))

        for pat in patnocancerlist:
            try:
                pat_res_f,pat_res_fc = preProcImage(pat, "Images_noBG")
                no_cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, no_cancer_results)
            except:
                print("Error")

        displayResults(cancer_results, "Cancer")
        displayResults(no_cancer_results, "No Cancer")
        # writeResultsToCsv(cancer_results, no_cancer_results, 'cancer_res_csv4.csv')

    # Retinex optional parameters
    # retinex(patient id, folder reading from, setTransform=False, writeToFolder=None)
    # setTransform=True : display plot of the change in the image after transformation
    # writeToFolder='C:/some/path' : if provided a path, it will write images to a desired folder
    elif arg_len == 2 and sys.argv[1] == "retinex":
        # Get patient id lists to retrieve image files
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        for pat in patlist:
            retinex(pat, "Images_noBG")

        for pat in patnocancerlist:
            retinex(pat, "Images_noBG")


    elif arg_len == 2 and sys.argv[1] == "noiseReduce":
        print(noiseReduce("EscMar261010","Images_noBG"))

    elif arg_len == 2 and sys.argv[1] == "singleImageHist":

        # Get first patient id from each set to display results if desired
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patlist = [patlist[0]]
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")
        patnocancerlist = [patnocancerlist[0]]

        # Optional paramters for singleImHist:
        # display=True to the singleImHist function to display histogram
        # limit=1000, sets the y_limit to the histogram
        for p_id in patlist:
            f_im = cv2.imread('./Images_noBG/'+p_id+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+p_id+'A2BA-fc.jpg',1)
            singleImHist(f_im, display=True,limit=1000)
            singleImHist(fc_im, display=True,limit=1000)


        for p_id in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+p_id+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+p_id+'A2BA-fc.jpg',1)
            singleImHist(f_im, display=True,limit=1000)
            singleImHist(fc_im, display=True,limit=1000)

    elif arg_len == 2 and sys.argv[1] == "totalImageHist":
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        r_t, g_t, b_t = np.array([]), np.array([]), np.array([])

        for pat in patlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            r_t, g_t, b_t = totalImHist(f_im,b_t,g_t,r_t)
            r_t, g_t, b_t = totalImHist(fc_im,b_t,g_t,r_t)

        showtotalImHist(b_t,g_t,r_t)
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")
        r_t, g_t, b_t = np.array([]), np.array([]), np.array([])

        for pat in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            r_t, g_t, b_t = totalImHist(f_im,b_t,g_t,r_t)
            r_t, g_t, b_t = totalImHist(fc_im,b_t,g_t,r_t)
        showtotalImHist(b_t,g_t,r_t)

    elif arg_len == 2 and sys.argv[1] == "imageToCsv":
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        for pat in patlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            imageToCSV(f_im, './Images_csv/Cancer/'+pat+'A2BA-f.csv')
            imageToCSV(fc_im,'./Images_csv/Cancer/'+pat+'A2BA-fc.csv')

        for pat in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            imageToCSV(f_im, './Images_csv/Non-Cancer/'+pat+'A2BA-f.csv')
            imageToCSV(fc_im, './Images_csv/Non-Cancer/'+pat+'A2BA-fc.csv')

    elif arg_len == 2 and sys.argv[1] == "csvToImage":
        csvToImage('AcoAlm221112','./Images_csv/Non-Cancer/')

if __name__ == '__main__':
    main()
