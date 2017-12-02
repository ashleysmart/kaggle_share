import numpy as np
import skimage.exposure

# https://www.kaggle.com/gabrielaltay/exploring-color-scaling-images/notebook
def equalize_channels(img):
    for chan in range(img.shape[2]):
        im = img[:,:,chan]
        im = skimage.exposure.equalize_adapthist(im)
        img[:,:,chan] = im

def stretch_n(bands, lower_percent=5, higher_percent=95):
    # out = np.zeros_like(bands)
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)

