from fAndrew import rgbProcData
from fNatalie import tempProcData
from carter import clf

def NCA(temp_csv, image):
    results_natalie = tempProcData(temp_csv)

    results_andrew = rgbProcData(im)

    total_proc_data = results_natalie + results_andrew


    clf.predict(image)

    return 0.5, {}
