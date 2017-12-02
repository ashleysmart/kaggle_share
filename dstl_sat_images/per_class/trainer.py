#!/usr/bin/env python

import libs.patch_generator
import unet_model1
import numpy as np

if __name__ == "__main__":
    # hype params to consider later
    patch_size = 160

    x = np.load("./prep/3band_in.npy")

    for j in range(30): # 30 master epoach
        for i in range(10):  # ten class
            chan = i + 1
            y = np.load("./prep/3band_chan%d.npy" % chan)

            gen = libs.patch_generator.DataGenerator(x,y,
                patch_size           = patch_size)

            model = unet_model1.UnetModel("unet_model/unet_chan%d" % chan, 
                patch_size=patch_size,
                in_chan=3,
                out_chan=1)

            model.load()

            if j == 0:
                # quick pre train..
                model.train(gen, fit_epochs=1, gen_epochs=1)
            else:
                model.train(gen, fit_epochs=10, gen_epochs=10)

            #model.