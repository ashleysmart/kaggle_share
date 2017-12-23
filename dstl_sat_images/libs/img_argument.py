import numpy as np
import skimage.exposure

# https://www.kaggle.com/gabrielaltay/exploring-color-scaling-images/notebook
def equalize_channels(img):
    out = np.zeros_like(img).astype(np.float32)
    for chan in range(img.shape[2]):
        im = img[:,:,chan]
        im = skimage.exposure.equalize_adapthist(im)
        out[:,:,chan] = im

    return out

def stretch_n(img, lower_percent=5, higher_percent=95):
    out = np.zeros_like(img).astype(np.float32)
    chan = img.shape[2]
    for i in range(chan):
        band = img[:,:,i]
        a = 0 
        b = 1 
        c = np.percentile(band, lower_percent)
        d = np.percentile(band, higher_percent)
        t = a + (band - c) * (b - a) / (d - c)
        
        #threashold 
        t[t < a] = a
        t[t > b] = b
        out[:,:,i] = t

    return out

