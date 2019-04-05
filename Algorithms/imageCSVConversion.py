import cv2
import numpy as np

def imageToCSV(image,filename):
    b,g,r = cv2.split(image)
    filename_b = filename[:-4] + 'b' + filename[-4:]
    filename_g = filename[:-4] + 'g' + filename[-4:]
    filename_r = filename[:-4] + 'r' + filename[-4:]
    print(filename_b,filename_g,filename_r)
    np.savetxt(filename_b,b,delimiter=',')
    np.savetxt(filename_g,g,delimiter=',')
    np.savetxt(filename_r,r,delimiter=',')

def csvToImage(patString,csv_folder):
    dir_path = csv_folder

    b = np.loadtxt(open(dir_path + patString+'A2BA-fb.csv',"rb"), delimiter=',')
    g = np.loadtxt(open(dir_path + patString+'A2BA-fg.csv',"rb"), delimiter=',')
    r = np.loadtxt(open(dir_path + patString+'A2BA-fr.csv',"rb"), delimiter=',')
    
    # Combine the r,g,b channels to recreate image
    im = cv2.merge([b,g,r])
    print(im.shape)
    return im
