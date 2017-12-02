#!/usr/bin/env python

import libs.patch_generator
import roi_model
import numpy as np

patch_size = {
        1 :  100, # 'Bldg',
        2 :  100, # 'Struct',
        3 :  50,  # 'Road',
        4 :  50,  # 'Track',
        5 :  50,  # 'Trees',
        6 :  100, # 'Crops',
        7 :  50,  # 'Fast H20',
        8 :  100, # 'Slow H20',
        9 :  50,  # 'Truck',
        10 : 50   # 'Car',
        }

if __name__ == "__main__":
    x = np.load("./prep/3band_in.npy")

    # TODO restartability (stating at 3 this time it cashed before)
    channels = (np.arange(10) + 0) % 10 + 1

    for j in range(30): # 30 master epoach
        for chan in channels.tolist():  # ten class        

            print "loading data...."        
            y = np.load("./prep/3band_chan%d.npy" % chan)

            print "creating geneator...."        
            gen = libs.patch_generator.DataGenerator(x,y,
                patch_size           = patch_size[chan],
                mode                 = libs.patch_generator.DataGenerator.ROI)

            print "loading model...."        
            model = roi_model.RoiModel("roi_model/roi_chan%d" % chan, 
                patch_size=patch_size[chan],
                in_chan=3,
                out_chan=1)

            model.load()

            print "training model...."        
            if j == 0:
                # quick pre train..
                model.train(gen, fit_epochs=10, gen_epochs=1, 
                    val_samples=9000, gen_samples=30000)
            else:
                model.train(gen, fit_epochs=50, gen_epochs=10, 
                    val_samples=15000, gen_samples=50000)

            # make certian memory clears on each cycle or it will rn out
            del model
            del gen