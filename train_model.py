import sys
import os
import numpy as np
import cv2
from Algorithms.rgbData import rgbProcData
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


def extractFeatures(f_im,fc_im):
    andrew_results = processPatient(f_im, fc_im)
    carter_results = np.array([0]*64)
    natalie_results = np.array([0]*10)
    return np.concatenate([andrew_results,carter_results,natalie_results])


def main():
    arg_len = len(sys.argv)
    cancer_list = open("./patientAssignments/patients_cancer.txt","r").readlines()
    no_cancer_list = open("./patientAssignments/patients_nocancer.txt","r").readlines()
    cancer_list = [p.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for p in cancer_list]
    no_cancer_list = [p.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for p in no_cancer_list]

    cancer_res, nocancer_res = [], []


    for pat in cancer_list:
        f_im, fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1), cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
        cancer_res.append(extractFeatures(f_im,fc_im))

    for pat in no_cancer_list:
        f_im, fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1), cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
        nocancer_res.append(extractFeatures(f_im,fc_im))


    if arg_len >= 2 and sys.argv[1] == 'train':

        # Prepare Input Data
        all_inputs = cancer_res + nocancer_res
        # Prepare Input Classes
        all_classes = ['Y']*len(cancer_list) + ['N']* len(no_cancer_list)
        (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)

        # Create Pipeline
        steps = [('SVM', SVC(probability=True))]
        pipeline = Pipeline(steps) # define the pipeline object.
        parameters = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01,0.001,0.0001]}
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        grid.fit(train_inputs, train_classes)
        print(grid.score(test_inputs,test_classes))

        if arg_len > 2:
            test_pats = sys.argv[2:]

            for pat in test_pats:
                try:
                    f_test, fc_test = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1), cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
                    X = extractFeatures(f_test,fc_test).reshape(1,-1)
                    print(pat, grid.predict_proba(X))
                except:
                    print("Patient Not Found: ",pat)

if __name__ == '__main__':
    main()
