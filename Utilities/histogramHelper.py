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

        if limit:
            ax1.set_ylim(0,limit)
            ax2.set_ylim(0,limit)
            ax3.set_ylim(0,limit)
        plt.show()

    return b,g,r

def totalImHist(patient_list, display=True, limit=None):
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

    if display:
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        fig.suptitle("Total Image Histogram")
        ax1.hist(b_total, bins=np.arange(256),color='blue', ec="blue")
        ax2.hist(g_total, bins=np.arange(256), color='green', ec="green")
        ax3.hist(r_total, bins=np.arange(256), color="red",ec="red")

        if limit:
            ax1.set_ylim(0,limit)
            ax2.set_ylim(0,limit)
            ax3.set_ylim(0,limit)

        plt.show()
    return b_total, g_total, r_total



def compareTransformHistChannelHelper(im_tumor,im_non_tumor, transform_tumor, transform_non_tumor, color,limit=500,pat_id=''):
    fig1, axs1 = plt.subplots(2,1)

    if pat_id:
        fig1.suptitle(pat_id)

    (ax1,ax2) = axs1.flatten()

    ax1.set_title('Image- '+color)
    ax1.hist([im_tumor,im_non_tumor],bins=np.arange(256),histtype='step', fill=False, color=[color,'black'], label=['Tumor', 'Non-Tumor'])
    plt.legend()

    ax2.set_title('Transform- '+color)
    ax2.hist([transform_tumor,transform_non_tumor],bins=np.arange(256),color=[color,'black'], label=['Tumor', 'Non-Tumor'], histtype='step', fill=False,)

    [x.set_xlim(right=256) for x in [ax1,ax2]]
    if limit:
        [x.set_ylim(0,limit) for x in [ax1,ax2]]


    plt.legend()
    plt.show()


# Param: image { numpy.ndarray } - image read by cv2.imread
# Param: transformed { numpy.ndarray } - image read by cv2.imread
# Param: side_of_tumor { string } - 'L', 'R', or 'N'
def compareTransformHist(image, transformed, side_of_tumor,display=True,limit=500,pat_id=None):
    _, im_width, _ = image.shape
    _, transform_width, _ = transformed.shape

    if side_of_tumor == 'L':
        # Set tumor to left side
        tumorImage = image[:,0:(im_width/2)]
        nonTumorImage = image[:,(im_width/2):im_width]
        tumorTransformed = transformed[:,0:(transform_width/2)]
        nonTumorTransformed = transformed[:,(transform_width/2):transform_width]
    else:
        # Set tumor to right side
        tumorImage = image[:,(im_width/2):im_width]
        nonTumorImage = image[:,0:(im_width/2)]
        tumorTransformed = transformed[:,(transform_width/2):transform_width]
        nonTumorTransformed = transformed[:,0:(transform_width/2)]
    
    b_tumor_im, g_tumor_im, r_tumor_im = cv2.split(tumorImage)
    b_non_tumor_im, g_non_tumor_im, r_non_tumor_im = cv2.split(nonTumorImage)
    b_tumor_transform, g_tumor_transform, r_tumor_transform = cv2.split(tumorTransformed)
    b_non_tumor_transform, g_non_tumor_transform, r_non_tumor_transform = cv2.split(nonTumorTransformed)

    b_tumor_im, g_tumor_im, r_tumor_im = b_tumor_im.flatten(), g_tumor_im.flatten(), r_tumor_im.flatten()
    b_non_tumor_im, g_non_tumor_im, r_non_tumor_im = b_non_tumor_im.flatten(), g_non_tumor_im.flatten(), r_non_tumor_im.flatten()
    b_tumor_transform, g_tumor_transform, r_tumor_transform = b_tumor_transform.flatten(), g_tumor_transform.flatten(), r_tumor_transform.flatten()
    b_non_tumor_transform, g_non_tumor_transform, r_non_tumor_transform = b_non_tumor_transform.flatten(), g_non_tumor_transform.flatten(), r_non_tumor_transform.flatten()

    if display:
        compareTransformHistChannelHelper(b_tumor_im,b_non_tumor_im,b_tumor_transform,b_non_tumor_transform,'blue',limit=limit,pat_id=pat_id)
        compareTransformHistChannelHelper(r_tumor_im,r_non_tumor_im,r_tumor_transform,r_non_tumor_transform,'red',limit=limit,pat_id=pat_id)
        compareTransformHistChannelHelper(g_tumor_im,g_non_tumor_im,g_tumor_transform,g_non_tumor_transform,'green',limit=limit,pat_id=pat_id)



    return []
