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

def displayResults(results, title):
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.set_title(title + " Skew Data")
    skew_data = [results["skews"]["r_f"], results["skews"]["r_fc"], results["skews"]["g_f"], results["skews"]["g_fc"], results["skews"]["b_f"], results["skews"]["b_fc"]]
    colors = ['red', 'crimson','green', 'yellow','blue', 'skyblue']
    n, bins, patches = plt.hist(skew_data, 10, color=colors)

    ax = fig.add_subplot(2,2,2)
    ax.set_title(title + " Kurtosis Data")
    kurt_data = [results["kurtosis"]["r_f"], results["kurtosis"]["r_fc"], results["kurtosis"]["g_f"], results["kurtosis"]["g_fc"], results["kurtosis"]["b_f"], results["kurtosis"]["b_fc"]]
    colors = ['red', 'crimson','green', 'yellow','blue', 'skyblue']
    n, bins, patches = plt.hist(kurt_data, 10, color=colors)

    ax = fig.add_subplot(2,2,3, projection='3d')
    ax.set_title("Image Means - F")
    xf_vals = results["means"]['total_f']["r"]
    yf_vals = results["means"]['total_f']["g"]
    zf_vals = results["means"]['total_f']["b"]

    ax.scatter(xf_vals, yf_vals, zf_vals)
    ax.set_xlabel('R Values')
    ax.set_ylabel('G Values')
    ax.set_zlabel('B Values')

    ax = fig.add_subplot(2,2,4, projection='3d')
    ax.set_title("Image Means - FC")
    xfc_vals = results["means"]['total_fc']["r"]
    yfc_vals = results["means"]['total_fc']["g"]
    zfc_vals = results["means"]['total_fc']["b"]

    ax.scatter(xfc_vals, yfc_vals, zfc_vals)
    ax.set_xlabel('R Values')
    ax.set_ylabel('G Values')
    ax.set_zlabel('B Values')
    plt.show()

def compileHistogramResults(single_im_f, single_im_fc, current_results):
    current_results["skews"]["r_fc"].append(single_im_fc['r_skew'])
    current_results["skews"]["r_f"].append(single_im_f['r_skew'])
    current_results["skews"]["b_fc"].append(single_im_fc['b_skew'])
    current_results["skews"]["b_f"].append(single_im_f['b_skew'])
    current_results["skews"]["g_fc"].append(single_im_fc['g_skew'])
    current_results["skews"]["g_f"].append(single_im_f['g_skew'])
    current_results["kurtosis"]["r_fc"].append(single_im_fc['r_kurtosis'])
    current_results["kurtosis"]["r_f"].append(single_im_f['r_kurtosis'])
    current_results["kurtosis"]["b_fc"].append(single_im_fc['b_kurtosis'])
    current_results["kurtosis"]["b_f"].append(single_im_f['b_kurtosis'])
    current_results["kurtosis"]["g_fc"].append(single_im_fc['g_kurtosis'])
    current_results["kurtosis"]["g_f"].append(single_im_f['g_kurtosis'])
    current_results["means"]["total_f"]["b"].append(single_im_f['image_mean'][0])
    current_results["means"]["total_f"]["g"].append(single_im_f['image_mean'][1])
    current_results["means"]["total_f"]["r"].append(single_im_f['image_mean'][2])
    current_results["means"]["total_fc"]["b"].append(single_im_fc['image_mean'][0])
    current_results["means"]["total_fc"]["g"].append(single_im_fc['image_mean'][1])
    current_results["means"]["total_fc"]["r"].append(single_im_fc['image_mean'][2])
    return current_results

