import sys
import os
import numpy as np
import cv2
from fAndrew import rgbProcData
from writeCsvData import writeResults
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Param: f_im { cv2 Image } - f version of the front facing patient image
# Param: fc_im { cv2 Image } - fc version of the front facing patient image
# Return: { List } - concatenation of the processing of f and fc image lists
def processPatient(f_im, fc_im):
    f_res, fc_res = rgbProcData(f_im), rgbProcData(fc_im)
    patient_results = f_res + fc_res
    return patient_results

def main():
    arg_len = len(sys.argv)
    cancer_list = open("./patientAssignments/patients_cancer.txt","r").readlines()
    no_cancer_list = open("./patientAssignments/patients_nocancer.txt","r").readlines()
    cancer_list = [p.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for p in cancer_list]
    no_cancer_list = [p.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for p in no_cancer_list]

    cancer_res = []
    nocancer_res = []


    for pat in cancer_list:

        f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
        fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)

        andrew_results = processPatient(f_im, fc_im)
        carter_results = np.array([0]*64)
        natalie_results = np.array([0]*10)
        cancer_label = np.array(['Y'])
        cancer_res.append(np.concatenate([andrew_results,carter_results,natalie_results,cancer_label]))

    for pat in no_cancer_list:
        f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
        fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)

        andrew_results = processPatient(f_im, fc_im)
        carter_results = np.array([0]*64)
        natalie_results = np.array([0]*10)
        cancer_label = np.array(['N'])

        nocancer_res.append(np.concatenate([andrew_results,carter_results,natalie_results,cancer_label]))

    if arg_len == 2 and sys.argv[1] == 'model':

        # Prepare Data
        all_results = cancer_res + nocancer_res
        all_inputs = [x[:-1] for x in all_results]
        all_classes = [x[-1] for x in all_results]
        (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7)

        # Create Pipeline
        steps = [('SVM', SVC())]
        pipeline = Pipeline(steps) # define the pipeline object.
        parameters = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01,0.001,0.0001]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        grid.fit(train_inputs, train_classes)
        print(grid.score(test_inputs,test_classes))
        print(grid.best_estimator_)

    elif arg_len == 3 and sys.argv[1] == 'write':
        writeResults(cancer_res,nocancer_res,sys.argv[2])

if __name__ == '__main__':
    main()
