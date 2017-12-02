#!/usr/bin/env python

# Note... 
# version 1 - merge alll data into massive image.. 
#  -- problem when you try to do image argumentation (rotate etc)
# vesion 2 - load all images and masks into on 13d array
#  -- rgb image set has massive size and all the masks kills memory.. 
# version 3 -load images and save to disk.. build mask set by mask set and save each to disk

# ideas
# float 32 is a waste... 8 bit then up to float for model??.. saves 3/4 memoy size 
# i should aslo add +/- 1/255 random noise to the numbers 


import numpy as np
import pandas as pd
import os

import libs.raw_loader
import libs.img_argument
import libs.mask_generator

def make_training_sets(train_wkt_list, grid_sizes):
    prepDir      = "./prep/"
    in_src_tag   = "3band"
    in_src_func  = libs.raw_loader.img_3band
    in_src_argm  = libs.img_argument.stretch_n
    in_channels  = 3
    out_channels = 10

    ids = sorted(train_wkt_list.ImageId.unique())

    # first load images... we are loading max size padding with zero (mask will be off for 0)
    shapes = []
    idents = []
    imgs   = []
    for i in range(len(ids)):
        ident = ids[i]
        img = in_src_func(ident)
        img = in_src_argm(img)
        idents.append(ident)
        imgs.append(img)
        shapes.append(img.shape)
        
        print "loaded:", ident, img.shape
                
    shapes = np.array(shapes)
    size = np.amax(shapes)
    shape = (len(idents), size, size, in_channels)
    print "input working shape:", shape

    # store inputs
    z = np.zeros(shape, dtype=np.float32)
    for i in range(len(idents)):
        ident = idents[i]
        img = imgs[i]        
        print "packing input:", ident, img.shape, " maxsize:", size
        ident = idents[i]
        z[i,:img.shape[0],:img.shape[1],:] = img
        
    np.save(prepDir + in_src_tag + "_in", z)
    del imgs
    del z
    
    # generate masks
    shape = (len(idents), size, size, 1)
    print "mask working shape:", shape
    for j in range(out_channels):  # output channels
        z = np.zeros(shape, dtype=np.float32)
        channel = j + 1
        print "channel:", channel

        for i in range(len(idents)):  # images
            print "masking:", ident, shape, " maxsize:", size
            ident = idents[i]
            shape = shapes[i]
            msk = libs.mask_generator.generate_mask_for_image_and_class(
                (shape[0], shape[1]), 
                 ident, channel,
                 grid_sizes, train_wkt_list)
            msk = msk.reshape(shape[0], shape[1], 1)
            z[i,:shape[0],:shape[1]] = msk

        np.save(prepDir + in_src_tag + "_chan" + str(channel), z)
        del z

if __name__ == "__main__":
    train_wkt_list   = pd.read_csv(os.path.join(dataDir, 'train_wkt_v4.csv'))
    grid_sizes       = pd.read_csv(os.path.join(dataDir, 'grid_sizes.csv'), 
        names=['ImageId', 'Xmax', 'Ymin'], 
        skiprows=1)
    
    make_training_sets(train_wkt_list, grid_sizes)