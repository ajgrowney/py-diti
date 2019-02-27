# Py-Diti
This repository is for the image processing portion of our early detection of
breast cancer using machine learning

## How to Use

### image_preproc.py
Command: python image_preproc.py 'results.csv'
Description: Uses the txt files to discover our cancerous vs non-cancerous
sets and then reads in patient results from rgbProcData

## Functions

### processPatient
Parameter: pat - string of patient to read in <br />
Parameter: folder - folder to look in for the patient's photo <br />
Return: List < Integer > - Data from the patients <br />
List[0 - 2]: skew of r,g,b <br />
List[3-5]: kurtosis of r,g,b <br />
List[6-8]: total image mean pixel r,g,b values <br />
List[9-11]: left half of image mean pixel r,g,b <br />
List[12-14]: right half of image mean pixel r,g,b <br />
List[15-29]: same but with fc version of image
