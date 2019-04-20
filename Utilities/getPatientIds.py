

def getPatients(patients_path):
    patlist = open(patients_path, "r").readlines()
    patlist = [pat.strip().decode('utf-8-sig').encode('utf-8').replace('\r','') for pat in patlist]
    return patlist
