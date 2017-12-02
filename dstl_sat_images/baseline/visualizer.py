#!/usr/bin/env python

# https://www.kaggle.com/amanbh/visualize-polygons-and-image-data/code

"""
Author : amanbh

- Set up some basic functions to load/manipulate image data
- Visualize/Summarize cType counts, training data, and true classes
- Plot Polygons with holes correctly by using descartes package

Based on Kernel by
    Author : Oleg Medvedev
    Link   : https://www.kaggle.com/torrinos/dstl-satellite-imagery-feature-detection/exploration-and-plotting/run/553107
"""

import pandas as pd
import numpy as np
import os

from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch

import matplotlib.pyplot as plt
import tifffile as tiff

import pylab
# turn interactive mode on so that plots immediately
# See: http://stackoverflow.com/questions/2130913/no-plot-window-in-matplotlib
# pylab.ion()

dataDir = './data'
predDir = './predictions'
outDir  = './visual'

# Give short names, sensible colors and zorders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '0.9',  # '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }

ZORDER = {
        1 : 5,
        2 : 6,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }

# read the training data from train_wkt_v4.csv
train_wkt_list      = pd.read_csv(os.path.join(dataDir, 'train_wkt_v4.csv'))
grid_sizes          = pd.read_csv(os.path.join(dataDir, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
submission_wkt_list = pd.read_csv(os.path.join(predDir, 'submission.csv'))

# imageIds in a DataFrame
allImageIds   = grid_sizes.ImageId.unique()
trainImageIds = train_wkt_list.ImageId.unique()
subImageIds   = submission_wkt_list.ImageId.unique()

def get_image_names(imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': '{}/three_band/{}.tif'.format(dataDir, imageId),
         'A': '{}/sixteen_band/{}_A.tif'.format(dataDir, imageId),
         'M': '{}/sixteen_band/{}_M.tif'.format(dataDir, imageId),
         'P': '{}/sixteen_band/{}_P.tif'.format(dataDir, imageId),
         }
    return d


def get_images(imageId, img_key = None):
    '''
    Load images correspoding to imageId

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    img_key : {None, '3', 'A', 'M', 'P'}, optional
        Specify this to load single image
        None loads all images and returns in a dict
        '3' loads image from three_band/
        'A' loads '_A' image from sixteen_band/
        'M' loads '_M' image from sixteen_band/
        'P' loads '_P' image from sixteen_band/

    Returns
    -------
    images : dict
        A dict of image data from TIFF files as numpy array
    '''
    img_names = get_image_names(imageId)
    images = dict()
    if img_key is None:
        for k in img_names.keys():
            images[k] = tiff.imread(img_names[k])
    else:
        images[img_key] = tiff.imread(img_names[img_key])
    return images


def get_size(imageId):
    """
    Get the grid size of the image

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    """
    xmax, ymin = grid_sizes[grid_sizes.ImageId == imageId].iloc[0,1:].astype(float)
    W, H = get_images(imageId, '3')['3'].shape[1:]
    return (xmax, ymin, W, H)


def is_training_image(imageId):
    return any(trainImageIds == imageId)

def is_submission_image(imageId):
    return any(subImageIds == imageId)

def plot_polygons(fig, ax, polygonsList):
    '''
    Plot descrates.PolygonPatch from list of polygons objs for each CLASS
    '''
    legend_patches = []
    for cType in polygonsList:
        print('{} : {} \tcount = {}'.format(cType, CLASSES[cType], len(polygonsList[cType])))
        legend_patches.append(Patch(color=COLORS[cType],
                                    label='{} ({})'.format(CLASSES[cType], len(polygonsList[cType]))))
        for polygon in polygonsList[cType]:
            mpl_poly = PolygonPatch(polygon,
                                    color=COLORS[cType],
                                    lw=0,
                                    alpha=0.7,
                                    zorder=ZORDER[cType])
            ax.add_patch(mpl_poly)
    # ax.relim()
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return legend_patches


def plot_image(fig, ax, imageId, img_key, selected_channels=None):
    '''
    Plot get_images(imageId)[image_key] on axis/fig
    Optional: select which channels of the image are used (used for sixteen_band/ images)
    Parameters
    ----------
    img_key : str, {'3', 'P', 'N', 'A'}
        See get_images for description.
    '''
    images = get_images(imageId, img_key)
    img = images[img_key]
    title_suffix = ''
    if selected_channels is not None:
        img = img[selected_channels]
        title_suffix = ' (' + ','.join([ repr(i) for i in selected_channels ]) + ')'
    if len(img.shape) == 2:
        new_img = np.zeros((3, img.shape[0], img.shape[1]))
        new_img[0] = img
        new_img[1] = img
        new_img[2] = img
        img = new_img
    
    tiff.imshow(img, figure=fig, subplot=ax)
    ax.set_title(imageId + ' - ' + img_key + title_suffix)
    ax.set_xlabel(img.shape[-2])
    ax.set_ylabel(img.shape[-1])
    ax.set_xticks([])
    ax.set_yticks([])



def visualize_image(imageId, plot_all=0):
    '''         
    Plot all images and object-polygons
    
    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    plot_all : bool, True by default
        If True, plots all images (sixteen_band) as subplots.
        Otherwise, plot 3 band/training/predictions
    '''         
    xmax, ymin, W, H = get_size(imageId)
    
    if plot_all:
        # just the core objects
        fig, axArr = plt.subplots(figsize=(20, 20), nrows=3, ncols=3)
        fstArr = axArr[0]
    else:
        fig, axArr = plt.subplots(figsize=(30, 10), nrows=1, ncols=3)
        fstArr = axArr

    print('Image : {}'.format(imageId))
    plot_image(fig, fstArr[0], imageId, '3')

    if is_training_image(imageId):
        class_wkt = train_wkt_list[train_wkt_list.ImageId == imageId]

        ax = fstArr[1]
        polygonsList = {}
        for cType in CLASSES.keys():
            polygonsList[cType] = wkt_loads(class_wkt[class_wkt.ClassType == cType].MultipolygonWKT.values[0])

        legend_patches = plot_polygons(fig, ax, polygonsList)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 0)

    if is_submission_image(imageId):
        class_wkt = submission_wkt_list[submission_wkt_list.ImageId == imageId]  
        ax =  fstArr[2]
        polygonsList = {}
        for cType in CLASSES.keys():
            polygonsList[cType] = wkt_loads(class_wkt[class_wkt.ClassType == cType].MultipolygonWKT.values[0])
        legend_patches = plot_polygons(fig, ax, polygonsList)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 0)
    
    if plot_all:
        plot_image(fig, axArr[0][2], imageId, 'P')
        plot_image(fig, axArr[1][0], imageId, 'A', [0, 3, 6])
        plot_image(fig, axArr[1][1], imageId, 'A', [1, 4, 7])
        plot_image(fig, axArr[1][2], imageId, 'A', [2, 5, 0])
        plot_image(fig, axArr[2][0], imageId, 'M', [0, 3, 6])
        plot_image(fig, axArr[2][1], imageId, 'M', [1, 4, 7])
        plot_image(fig, axArr[2][2], imageId, 'M', [2, 5, 0])

    ax.legend(handles=legend_patches,
               # loc='upper center',
               bbox_to_anchor=(0.9, 1),
               bbox_transform=plt.gcf().transFigure,
               ncol=5,
               fontsize='x-small',
               title='Objects-' + imageId,
               # mode="expand",
               framealpha=0.3)

    return (fig, axArr, ax)

# Loop over few training images and save to files
#for imageId in trainImageIds:
for imageId in allImageIds:
    fig, axArr, ax = visualize_image(imageId, plot_all=False)
    plt.savefig(outDir + '/' + imageId + '_vis.png')
    plt.clf()
