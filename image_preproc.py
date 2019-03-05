import sys
import numpy as np
import cv2
from fAndrew import rgbProcData
from writeCsvData import writeResults


# Param: f_im { cv2 Image } - f version of the front facing patient image
# Param: fc_im { cv2 Image } - fc version of the front facing patient image
# Return: { List } - concatentation of the processing of f and fc image lists
def processPatient(f_im, fc_im):
    f_res, fc_res = rgbProcData(f_im), rgbProcData(fc_im)
    patient_results = f_res + fc_res
    return patient_results

def main():
    arg_len = len(sys.argv)
    cancer_list = open("./patientAssignments/patients_cancer.txt","r").readlines()
    no_cancer_list = open("./patientAssignments/patients_nocancer.txt","r").readlines()
    cancer_list = [p.strip().replace('\r','') for p in cancer_list]
    no_cancer_list = [p.strip().replace('\r','') for p in no_cancer_list]

    cancer_res = []
    nocancer_res = []

    for pat in cancer_list:
        try:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            cancer_res.append(processPatient(f_im, fc_im))
        except AttributeError as e:
            print("Patient Not Found: ",pat)

    for pat in no_cancer_list:
        try:
            f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
            fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
            nocancer_res.append(processPatient(f_im, fc_im))
        except AttributeError as e:
            print("Patient Not Found: ",pat)


    if arg_len == 2:
        writeResults(cancer_res,nocancer_res,sys.argv[1])

if __name__ == '__main__':
    main()
