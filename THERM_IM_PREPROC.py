import sys
import statistics
import json
import csv
import numpy as np
import pandas as pd
import cv2
#import colorcorrect.algorithm as cca
from scipy.misc import imread
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA, PCA

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
    b,g,r = cv2.split(f_im)
    f_im = cv2.merge([r,g,b])
    fc_im = cv2.imread('./'+folder+'/'+image+'A2BA-fc.jpg',1)
    b,g,r = cv2.split(fc_im)
    fc_im = cv2.merge([r,g,b])


    # Utilize the RGB Colorspace to analyze the image and pull statistics
    rgb_proc_f = rgbProcData(f_im, image)
    rgb_proc_fc = rgbProcData(fc_im,image)

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


def retinex(patString, folder):
    f_im = cv2.imread('./'+folder+'/'+patString+'A2BA-f.jpg',-1)
    # b,g,r = cv2.split(f_im)
    # f_im = cv2.merge([r,g,b])
    fc_im = cv2.imread('./'+folder+'/'+patString+'A2BA-fc.jpg',-1)
    # b,g,r = cv2.split(fc_im)
    # fc_im = cv2.merge([r,g,b])
    a = cca.retinex(f_im)
    b = cca.retinex(fc_im)
    # plt.subplot(4,1,1), plt.imshow(a)
    # plt.subplot(4,1,2), plt.imshow(f_im)
    # plt.subplot(4,1,3), plt.imshow(b)
    # plt.subplot(4,1,4), plt.imshow(fc_im)
    # plt.show()
    cv2.imwrite('./Images_retinex/'+patString+'A2BA-f.jpg',a)
    cv2.imwrite('./Images_retinex/'+patString+'A2BA-fc.jpg',b)
    return []

def singleImHist(im):
    b,g,r = cv2.split(im)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    # Plot the histograms below
    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    # fig.suptitle("Single Image Histogram")
    # ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
    # ax1.set_ylim(0,1500)
    # ax2.hist(g, bins=np.arange(256), color='green', ec="green")
    # ax2.set_ylim(0,1500)
    # ax3.hist(r, bins=np.arange(256), color="red",ec="red")
    # ax3.set_ylim(0,1500)
    # plt.show()
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

def showtotalImHist(b,g,r):
    # Plot the histograms below
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.suptitle("Total Image Histogram")
    ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
    # ax1.set_ylim(0,1500)
    ax2.hist(g, bins=np.arange(256), color='green', ec="green")
    # ax2.set_ylim(0,1500)
    ax3.hist(r, bins=np.arange(256), color="red",ec="red")
    # ax3.set_ylim(0,1500)
    plt.show()

def imageToCSV(image,filename):
    b,g,r = cv2.split(image)
    filename_b = filename[:-4] + 'b' + filename[-4:]
    filename_g = filename[:-4] + 'g' + filename[-4:]
    filename_r = filename[:-4] + 'r' + filename[-4:]
    np.savetxt(filename_b,b,delimiter=',')
    np.savetxt(filename_g,g,delimiter=',')
    np.savetxt(filename_r,r,delimiter=',')
        
def csvToImage(patString,cancer):
    dir_path = "./"
    if cancer == 'Y':
        dir_path = './Images_csv/Cancer/'
    elif cancer == 'N':
        dir_path = './Images_csv/Non-Cancer/'
    b = np.loadtxt(open(dir_path + patString+'A2BA-fb.csv',"rb"), delimiter=',')
    g = np.loadtxt(open(dir_path + patString+'A2BA-fg.csv',"rb"), delimiter=',')
    r = np.loadtxt(open(dir_path + patString+'A2BA-fr.csv',"rb"), delimiter=',')
    im = np.dstack([b,g,r])
    return im



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

        patlist = open("./patientAssignments/patients_cancer.txt", "r").readlines()
        patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
        for pat in patlist:
            try:
                pat_res_f, pat_res_fc = preProcImage(pat, "Images_noBG")
                cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, cancer_results)
            except:
                print("Error", pat.strip().replace('\r',''))
        # displayResults(cancer_results, "Cancer")

        patnocancerlist = open("./patientAssignments/patients_nocancer.txt", "r").readlines()
        patnocancerlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patnocancerlist]

        print(len(patnocancerlist))
        for pat in patnocancerlist:
            try:
                pat_res_f,pat_res_fc = preProcImage(pat, "Images_noBG")
                no_cancer_results = compileHistogramResults(pat_res_f, pat_res_fc, no_cancer_results)
            except:
                print("Error")
        # displayResults(no_cancer_results, "No Cancer")
        # writeResultsToCsv(cancer_results, no_cancer_results, 'cancer_res_csv4.csv')

    elif arg_len == 2 and sys.argv[1] == "retinex":
        patlist = open("./patientAssignments/patients_cancer.txt", "r").readlines()
        patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
        for pat in patlist:
            retinex(pat, "Images_noBG")

        patnocancerlist = open("./patientAssignments/patients_nocancer.txt", "r").readlines()
        patnocancerlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patnocancerlist]

        for pat in patnocancerlist:
            retinex(pat, "Images_noBG")

    elif arg_len == 2 and sys.argv[1] == "noiseReduce":
        print(noiseReduce("EscMar261010","Images_noBG"))

    elif arg_len == 2 and sys.argv[1] == "singleImageHist":
        patlist = open("./patientAssignments/patients_cancer.txt", "r").readlines()
        patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
        patlist = [patlist[0]]
        for pat in patlist:
            print('./Images_noBG/'+pat+'A2BA-f.jpg')
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            singleImHist(f_im)
            singleImHist(fc_im)

        patnocancerlist = open("./patientAssignments/patients_nocancer.txt", "r").readlines()
        patnocancerlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patnocancerlist]
        patnocancerlist = [patnocancerlist[0]]

        for pat in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            singleImHist(f_im)
            singleImHist(fc_im)

    elif arg_len == 2 and sys.argv[1] == "totalImageHist":
        patlist = open("./patientAssignments/patients_cancer.txt", "r").readlines()
        patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
        patlist = [patlist[0]]
        r_t, g_t, b_t = np.array([]), np.array([]), np.array([])

        for pat in patlist:
            print('./Images_noBG/'+pat+'A2BA-f.jpg')
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            r_t, g_t, b_t = totalImHist(f_im,b_t,g_t,r_t)
            r_t, g_t, b_t = totalImHist(fc_im,b_t,g_t,r_t)

        showtotalImHist(b_t,g_t,r_t)
        patnocancerlist = open("./patientAssignments/patients_nocancer.txt", "r").readlines()
        patnocancerlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patnocancerlist]
        patnocancerlist = [patnocancerlist[0]]
        r_t, g_t, b_t = np.array([]), np.array([]), np.array([])

        for pat in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            r_t, g_t, b_t = totalImHist(f_im,b_t,g_t,r_t)
            r_t, g_t, b_t = totalImHist(fc_im,b_t,g_t,r_t)
        showtotalImHist(b_t,g_t,r_t)
    elif arg_len == 2 and sys.argv[1] == "imageToCsv":
        patlist = open("./patientAssignments/patients_cancer.txt", "r").readlines()
        patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
        for pat in patlist:
            print(pat+'A2BA-f.csv')
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            imageToCSV(f_im, './Images_csv/Cancer/'+pat+'A2BA-f.csv')
            imageToCSV(fc_im,'./Images_csv/Cancer/'+pat+'A2BA-fc.csv')

        patnocancerlist = open("./patientAssignments/patients_nocancer.txt", "r").readlines()
        patnocancerlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patnocancerlist]

        for pat in patnocancerlist:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            imageToCSV(f_im, './Images_csv/Non-Cancer/'+pat+'A2BA-f.csv')
            imageToCSV(fc_im, './Images_csv/Non-Cancer/'+pat+'A2BA-fc.csv')
    elif arg_len == 2 and sys.argv[1] == "csvToImage":
        csvToImage('AcoAlm221112','N')
if __name__ == '__main__':
    main()
