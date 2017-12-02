#!/usr/bin/env python

import libs.raw_loader
import libs.img_argument
import libs.patch_sequancer
import unet_model1

import numpy as np
import pandas as pd
import os

from collections import defaultdict

import cv2

import shapely.wkt
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon

dataDir = './data'

# loads test data, prepares it and applies the trained models on the data
def load_prepare_image(img_id):
    img = libs.raw_loader.img_3band(img_id)
    img = libs.img_argument.stretch_n(img)

    return img 

def get_scalers(im_size, x_max, y_min):
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h = float(im_size[0])
    w = float(im_size[1])
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    threashold_mask = ((mask == 1) * 255).astype(np.uint8)

    # opencv 3 
    # image, contours, hierarchy = cv2.findContours(threashold_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours, hierarchy = cv2.findContours(threashold_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def make_polygons(img_mask, x_max, y_min):
    x_scaler, y_scaler = get_scalers(img_mask.shape, x_max, y_min)
    pred_polygons = mask_to_polygons(img_mask)
    scaled_pred_polygons = shapely.affinity.scale(pred_polygons, 
            xfact=1.0 / x_scaler, 
            yfact=1.0 / y_scaler,
            origin=(0.0, 0.0, 0.0))

    return shapely.wkt.dumps(scaled_pred_polygons)

if __name__ == "__main__":
    # ok.. a note here i want to ovelap outputs so this can be un multiple times
    train_wkt_list      = pd.read_csv(os.path.join(dataDir, 'train_wkt_v4.csv'))
    grid_sizes          = pd.read_csv(os.path.join(dataDir, 'grid_sizes.csv'),
             names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    ofile = "predictions/polygons.csv"
    if os.path.isfile(ofile):
        submission_wkt_list = pd.read_csv(ofile)  # mine or all outputs
    else:
        submission_wkt_list = pd.read_csv(os.path.join(dataDir, 'all_results.csv'))  # mine or all outputs

    # hype params to consider later
    patch_size = 160
    patch_edge =  5    # +/- 5 pixels to interpolate prediction edges  
    patcher = libs.patch_sequancer.PatchInterpolationSequancer(patch_size,patch_edge,3,1)

    for i in range(10):  # all ten class
        chan = i + 1        
        model = unet_model1.UnetModel("unet_model/unet_chan%d" % chan, 
            patch_size=patch_size,
            in_chan=3,
            out_chan=1)
        model.load()

        # for img_id in sorted(set(submission_wkt_list['ImageId'].tolist())):
        for img_id in sorted(set(train_wkt_list['ImageId'].tolist())):
            cache_file = 'predictions/mask_%s_chan_%d' % (img_id, chan)

            if os.path.isfile(cache_file + ".npy"):
                continue

            print "processing", img_id, " for chan:", chan
                
            img = load_prepare_image(img_id) 
            #img = img.reshape((1,) + img.shape)
            # compte and merge model outputs

            mask = patcher(img, 1, lambda patch: model.predict(patch))

            np.save(cache_file, mask)

            # threshold merged results
            model.threshold(mask)
    
            # convet to ploygon
            idx = grid_sizes['ImageId'] == img_id
            x_max = grid_sizes.loc[idx, 'Xmax'].as_matrix()[0]
            y_min = grid_sizes.loc[idx, 'Ymin'].as_matrix()[0]
            polygon = make_polygons(mask, x_max, y_min)

            # save into results
            idx = (submission_wkt_list["ImageId"] == img_id) & (submission_wkt_list["ClassType"] == chan)
            submission_wkt_list.loc[idx, "MultipolygonWKT"] = polygon

            submission_wkt_list.to_csv(ofile, index=False)



