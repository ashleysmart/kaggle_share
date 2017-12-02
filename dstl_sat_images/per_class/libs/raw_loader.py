import tifffile as tiff
import numpy as np
import os

dataDir = './data'

def img_Aband(image_id):
    filename = os.path.join(dataDir, 'sixteen_band', '{}_A.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def img_Mband(image_id):
    filename = os.path.join(dataDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

def img_Pband(image_id):
    filename = os.path.join(dataDir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img = tiff.imread(filename)
    return img

def img_3band(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(dataDir, 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img