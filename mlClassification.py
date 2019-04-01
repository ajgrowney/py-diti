from sklearn.svm import SVC
from fAndrew import rgbProcData
from fNatalie import tempProcData
from fCarter import infer
import numpy as np
import pickle


def csvToImage(csv):
    return []


def loadClassifier():
    clf = pickle.load(open('final_model.sav','rb'))
    return clf


# C = 10, gamma = .0001
def NCA(temp_csv):
    # Conver CSV to Image
    image = csvToImage(temp_csv)

    # Calculate Pre-Processing Statistics
    results_natalie = tempProcData(temp_csv)
    results_andrew = rgbProcData(image)
    results_carter = infer(image)

    # Combine statistics
    total_proc_data = [results_natalie + results_andrew + results_carter]


    classifier = loadClassifier()
    print(type(classifier))
    res = classifier.predict(total_proc_data)
    print(res)
    return res, {}


NCA(np.array([]))
