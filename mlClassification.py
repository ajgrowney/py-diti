from sklearn.svm import SVC
from fAndrew import rgbProcData
from fNatalie import tempProcData
from fCarter import infer
import numpy as np
import pickle
import cv2


def csvToImage(csv):
    return []


def loadClassifier():
    clf = pickle.load(open('final_model.sav','rb'))
    return clf


def NCATwoImage(im_f,im_fc):


    # Calculate Pre-Processing Statistics
    results_andrew = rgbProcData(im_f,im_fc)
    results_carter = infer([im_f,im_fc])

    # Combine statistics
    total_proc_data = np.concatenate([results_andrew,results_carter])


    classifier = loadClassifier()
    print(type(classifier))
    res = classifier.predict(total_proc_data.reshape(1,-1))
    print(res)
    return res, {}


def NCAOneImage(full_im):


    # Calculate Pre-Processing Statistics
    results_andrew = rgbProcData(full_im)
    results_carter = infer([full_im])[0]

    # Combine statistics
    total_proc_data = np.concatenate([results_andrew,results_carter])


    classifier = loadClassifier()
    print(type(classifier))
    res = classifier.predict(total_proc_data.reshape(1,-1))
    print(res)
    return res, {}

test_im = cv2.imread('./Images_noBG/AcoAlm221112A2BA-f.jpg')
NCA(test_im)
