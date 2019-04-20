import sys
import statistics
import json
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
from Utilities.histogramHelper import singleImHist, totalImHist, compareTransformHist
from Utilities.resultsDevObj import resultsObj, writeResultsToCsv

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


def getPatients(patients_path):
    patlist = open(patients_path, "r").readlines()
    patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
    return patlist


def main():
    arg_len = len(sys.argv)

    if sys.argv[1] == "frontDataAll":
        # Results Object
        cancer_results = resultsObj()
        no_cancer_results = resultsObj()

        # Get patient id lists to retrieve image files
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        for pat in patlist:
            try:
                pat_res_f, pat_res_fc = preProcImage(pat, "Images_noBG")
                cancer_results.compileHistogramResults(pat_res_f, pat_res_fc)
            except Exception as e:
                print("Error", e)

        for pat in patnocancerlist:
            try:
                pat_res_f,pat_res_fc = preProcImage(pat, "Images_noBG")
                no_cancer_results.compileHistogramResults(pat_res_f, pat_res_fc)
            except Exception as e:
                print("Error",e.message)

        writeResultsToCsv(cancer_results,no_cancer_results,'./full_data.csv')
        #cancer_results.displayResults("Cancer")
        #no_cancer_results.displayResults("No Cancer")

    elif sys.argv[1] == "compareTransform":
        im0 = cv2.imread('./Images_noBG/AguCat100309A2BA-f.jpg')
        imt = cv2.imread('./Images_retinex/AguCat100309A2BA-f.jpg')
        compareTransformHist(im0,imt,side_of_tumor='L',display=True,limit=500,pat_id='AguCat100309')

    # Retinex optional parameters
    # retinex(patient_id, images_folder, setTransform=False, writeToFolder=None)
    # setTransform=True : display plot of the change in the image after transformation
    # writeToFolder='C:/some/path' : if provided a path, it will write images to a desired folder
    elif sys.argv[1] == "retinex":
        # Get patient id lists to retrieve image files
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        for pat in patlist:
            retinex(pat, "Images_noBG", writeToFolder='./Images_retinex/')
        
        for pat in patnocancerlist:
            retinex(pat, "Images_noBG", writeToFolder='./Images_retinex/')


    elif sys.argv[1] == "noiseReduce":
        print(noiseReduce("EscMar261010","Images_noBG"))

    elif sys.argv[1] == "singleImageHist":

        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        # Get first patient id from each set to display results if desired
        patlist = [patlist[0]]
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

    elif sys.argv[1] == "totalImageHist":
        patlist = getPatients("./patientAssignments/patients_cancer.txt")
        patnocancerlist = getPatients("./patientAssignments/patients_nocancer.txt")

        totalImHist(patlist, display=True)
        totalImHist(patnocancerlist, display=True)

    elif sys.argv[1] == "imageToCsv":
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

    elif sys.argv[1] == "csvToImage":
        pat_id = sys.argv[2] if len(sys.argv) > 2 else 'AcoAlm221112'
        pat_path = sys.argv[3] if len(sys.argv) > 3 else './Images_csv/Non-Cancer/'
        csvToImage(pat_id,pat_path)

if __name__ == '__main__':
    main()
