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

def totalImHist(im,b_total,g_total,r_total):
    b,g,r = cv2.split(im)
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    b_total = np.concatenate([b_total,b])
    g_total = np.concatenate([g_total,g])
    r_total = np.concatenate([r_total,r])
    return b_total, g_total, r_total

def showtotalImHist(b,g,r,limit=None):
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
