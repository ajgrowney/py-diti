# Py-Diti
This repository is for the image processing portion of our early detection of
breast cancer using machine learning

## How to Use


### image_preproc.py
Command: python image_preproc.py 'results.csv'
Description: Uses the txt files from './patientsAssignments/' folder to discover our cancerous vs non-cancerous
sets. After splitting into the sets, it reads the processPatient function data into result arrays.<br />
(If desired) You can enter a filename like 'results.csv' in the command to save the output to a csv file.

### fAndrew.py
Description: Standalone function built to return a vector of image metadata for our ensemble classifier. The input parameter should be an image file, preferably read in by OpenCV's cv2.imread function. The function computes a series of statistics documented below.

## Functions

### rgbProcData from fAndrew.py
Parameter: im - image file read in by cv2.imread(img,1) <br />
Return: List < Integer > - Data from the patient <br />
List[0 - 2]: skew of r,g,b <br />
List[3-5]: kurtosis of r,g,b <br />
List[6-8]: total image mean pixel r,g,b values <br />
List[9-11]: left half of image mean pixel r,g,b <br />
List[12-14]: right half of image mean pixel r,g,b <br />

### processPatient from image_preproc.py
Parameter: f_im { cv2Image } - f version of the front facing patient image <br />
Parameter: fc_im { cv2 Image } - fc version of the front facing patient image <br />
Return: { List } - concatenation of the processing of f and fc image lists <br />
