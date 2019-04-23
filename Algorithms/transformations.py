import cv2
import os
import matplotlib.pyplot as plt
# import colorcorrect.algorithm as cca


def noiseReduce(image, folder):
    f_im = cv2.imread('./'+folder+'/'+image+'A2BA-f.jpg',1)
    fc_im = cv2.imread('./'+folder+'/'+image+'A2BA-fc.jpg',0)

    im_b = f_im.copy()
    im_b[:, :, 1] = 0
    im_b[:, :, 2] = 0

    im_g = f_im.copy()
    im_g[:, :, 0] = 0
    im_g[:, :, 2] = 0

    im_r = f_im.copy()
    im_r[:, :, 0] = 0 # b -> 0
    im_r[:, :, 1] = 0 # g -> 0

    f_imb = cv2.cvtColor(im_b,cv2.COLOR_BGR2GRAY)
    f_img = cv2.cvtColor(im_g,cv2.COLOR_BGR2GRAY)
    f_imr = cv2.cvtColor(im_r,cv2.COLOR_BGR2GRAY)


    f_imb = cv2.medianBlur(f_imb,5)
    f_img = cv2.medianBlur(f_img,5)
    f_imr = cv2.medianBlur(f_imr,5)
    edges = cv2.Canny(f_imr,100,200)

    plt.subplot(2,2,1), plt.imshow(edges)
    plt.subplot(2,2,2), plt.imshow(f_imb)
    plt.subplot(2,2,3), plt.imshow(f_img)
    plt.subplot(2,2,4), plt.imshow(f_imr)

    plt.show()

def retinex(patString, folder, showTransform=False, writeToFolder=None):
    f_im = cv2.imread('./'+folder+'/'+patString+'A2BA-f.jpg',1)
    fc_im = cv2.imread('./'+folder+'/'+patString+'A2BA-fc.jpg',1)

    f_retinex = cca.retinex(f_im)
    fc_retinex = cca.retinex(fc_im)

    if showTransform:
        plt.subplot(4,1,1), plt.imshow(f_retinex)
        plt.subplot(4,1,2), plt.imshow(f_im)
        plt.subplot(4,1,3), plt.imshow(fc_retinex)
        plt.subplot(4,1,4), plt.imshow(fc_im)
        plt.show()

    if writeToFolder != None:
        filename_f = patString+'A2BA-f.jpg'
        filename_fc = patString+'A2BA-fc.jpg'
        writePath_f = os.path.join(writeToFolder,filename_f)
        writePath_fc = os.path.join(writeToFolder,filename_fc)
        cv2.imwrite(writePath_f,f_retinex)
        cv2.imwrite(writePath_fc,fc_retinex)
        
    return f_retinex, fc_retinex
