import matplotlib.pyplot as plt
import cv2
import numpy as np

def singleImHist(im, display=False, limit=None):
    b,g,r = cv2.split(im)

    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    if display:
        # Plot the histograms below
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        fig.suptitle("Single Image Histogram")
        ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
        ax2.hist(g, bins=np.arange(256), color='green', ec="green")
        ax3.hist(r, bins=np.arange(256), color="red",ec="red")

        if limit != None:
            ax1.set_ylim(0,limit)
            ax2.set_ylim(0,limit)
            ax3.set_ylim(0,limit)
        plt.show()

    return b,g,r

def totalImHist(patient_list):
    b_total,g_total,r_total = np.array([]), np.array([]), np.array([])
    for pat in patient_list:
        f_im = cv2.imread('./Images_noBG/'+pat+'A2BA-f.jpg',1)
        fc_im = cv2.imread('./Images_noBG/'+pat+'A2BA-fc.jpg',1)
        fb,fg,fr = cv2.split(f_im)
        fcb,fcg,fcr = cv2.split(fc_im)
        fb, fg, fr = fb.flatten(), fg.flatten(), fr.flatten()
        fcb, fcg, fcr = fcb.flatten(), fcg.flatten(), fcr.flatten()
        
        b_total = np.concatenate([b_total,fb,fcb])
        g_total = np.concatenate([g_total,fg,fcg])
        r_total = np.concatenate([r_total,fr,fcr])
    return b_total, g_total, r_total

def showtotalImHist(b,g,r,limit=None,display=True):
    if display == False:
        print b,g,r
        return
    
    # Plot the histograms below
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.suptitle("Total Image Histogram")
    ax1.hist(b, bins=np.arange(256),color='blue', ec="blue")
    ax2.hist(g, bins=np.arange(256), color='green', ec="green")
    ax3.hist(r, bins=np.arange(256), color="red",ec="red")

    if limit != None:
        ax1.set_ylim(0,limit)
        ax2.set_ylim(0,limit)
        ax3.set_ylim(0,limit)

    plt.show()
