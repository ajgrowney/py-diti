import sys
import numpy as np
import csv
from scipy.stats import kurtosis,skew
import cv2

def writeResults(cancer_res, nocancer_res, filename):
    f = open(filename,'w')
    with f:
        fieldnames_f = ['f_r_skew', 'f_g_skew', 'f_b_skew', 'f_r_kurt', 'f_g_kurt', 'f_b_kurt', 'f_r_image_mean', 'f_g_image_mean','f_b_image_mean', 'f_r_left_mean', 'f_g_left_mean', 'f_b_left_mean', 'f_r_right_mean', 'f_g_right_mean', 'f_b_right_mean']
        fieldnames_fc = ['fc_r_skew', 'fc_g_skew', 'fc_b_skew', 'fc_r_kurt', 'fc_g_kurt', 'fc_b_kurt', 'fc_r_image_mean', 'fc_g_image_mean','fc_b_image_mean', 'fc_r_left_mean', 'fc_g_left_mean', 'fc_b_left_mean', 'fc_r_right_mean', 'fc_g_right_mean', 'fc_b_right_mean']
        fieldnames = fieldnames_f + fieldnames_fc + ['cancer']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(cancer_res)):
            writer.writerow(cancer_res[i]+['Y'])
        for i in range(len(nocancer_res)):
            writer.writerow(nocancer_res[i]+['N'])


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

    b_skew, g_skew, r_skew = map(lambda x: skew(x), [b,g,r])
    b_kurt, g_kurt, r_kurt = map(lambda x: kurtosis(x), [b,g,r])
    
    results = [r_skew, g_skew, b_skew, r_kurt, g_kurt, b_kurt, image_mean[0], image_mean[1], image_mean[2], left_mean[0], left_mean[1], left_mean[2], right_mean[0], right_mean[1], right_mean[2]]
    return results

# Param: pat { String } - name of the patients to read images of
# Return: { List } - concatentation of the processing of f and fc image lists
def processPatient(pat, folder):
    f_im = cv2.imread('./'+folder+'/'+pat+'A2BA-f.jpg',1)
    fc_im = cv2.imread('./'+folder+'/'+pat+'A2BA-fc.jpg',1)
    f_res, fc_res = map(lambda x: rgbProcData(x), [f_im, fc_im])
    patient_results = f_res + fc_res
    return patient_results
    
def main():
    arg_len = len(sys.argv)
    cancer_list = open("patients_cancer.txt","r").readlines()
    no_cancer_list = open("patients_nocancer.txt","r").readlines()
    if arg_len == 2:
        cancer_res = []
        nocancer_res = []
        for pat in cancer_list:
            try:
                pat = pat.strip().replace('\r','')
                cancer_res.append(processPatient(pat, "Cancer_noBG"))
            except AttributeError as e:
                print("Patient Not Found: ",pat)
        
        for pat in no_cancer_list:
            try:
                pat = pat.strip().replace('\r','')
                nocancer_res.append(processPatient(pat, "Volunteer_noBG"))
            except AttributeError as e:
                print("Patient Not Found: ",pat)
        
        writeResults(cancer_res,nocancer_res,sys.argv[1])

if __name__ == '__main__':
    main()
