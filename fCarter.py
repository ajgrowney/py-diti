import numpy as np
def infer(list_im,im_metadata, temp_metadata):
    print(len(list_im))
    res = np.random.rand(len(list_im),64)
    res = res.tolist()
    return res
