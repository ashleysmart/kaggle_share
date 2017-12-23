#!/usr/bin/env python

# Note... 
# version 1 - merge alll data into massive image.. 
#  - problem when you try to do image argumentation (rotate etc)
# vesion 2 - load all images and masks into on 13d array
#  - rgb image set has massive size and all the masks kills memory.. 
# version 3
#  - load images and save to disk.. build mask set by mask set and save each to disk
# version 4 -- THIS FILE!
#  - channnels first
#   -- see notes in README.txt about GPU optimisation
#  - data type reduction
#   -- boolean masks
#   -- 8 bit color
#   -- note the intention is that later data argumentation will return the data to floats
#  - per image/per mask set files..
#   -- this is due to a road block (96% memory usage) with i tried to load all masks together
#   -- this allows choice later, load smaller parts of intereast
#   -- do image selection as random choice from file names

import numpy as np
import pandas as pd
import os

import libs.raw_loader
import libs.img_argument
import libs.mask_generator

def make_training_sets(train_wkt_list, grid_sizes):
    prepDir      = "./raw/"
    in_src_tag   = "3band"
    in_src_func  = libs.raw_loader.img_3band
    in_src_argm  = libs.img_argument.stretch_n
    in_channels  = 3
    out_channels = 10

    if not os.path.exists(prepDir):
        os.makedirs(prepDir)
    
    idents = sorted(train_wkt_list.ImageId.unique())

    # meh u can get idents from the dir listing or wkts file.. but saving it makes it easy later
    # infact this entire process can be done on the fly from inside a data_flows class

    np.save(prepDir + "idents.npy", np.array(idents))

    shapes = []
    
    # store inputs
    for i in range(len(idents)):
        ident = idents[i]

        print "loading input:", ident,
        
        img = in_src_func(ident)
        img = in_src_argm(img)
        img = (img * 255).astype(np.uint8)

        shapes.append(img.shape)
        print "packing input:", img.shape
        
        np.savez_compressed(prepDir + ident + "_" + in_src_tag + "_in.npz", img)

    shapes = np.array(shapes)

    # generate masks
    for i in range(len(idents)):  # images
        height, width, _ = shapes[i]
        print "masking:", ident, " shape:", height, width
        ident = idents[i]

        for j in range(out_channels):  # output channels
            channel = j + 1
            print " doing channel:", channel

            msk = libs.mask_generator.generate_mask_for_image_and_class(
                (height, width), 
                 ident, channel,
                 grid_sizes, train_wkt_list)

            np.savez_compressed(prepDir + ident + "_" + in_src_tag + "_chan" + str(channel) + ".npz", msk > 0.5)

if __name__ == "__main__":
    dataDir = "./data/"
    train_wkt_list   = pd.read_csv(os.path.join(dataDir, 'train_wkt_v4.csv'))
    grid_sizes       = pd.read_csv(os.path.join(dataDir, 'grid_sizes.csv'), 
        names=['ImageId', 'Xmax', 'Ymin'], 
        skiprows=1)
    
    make_training_sets(train_wkt_list, grid_sizes)
