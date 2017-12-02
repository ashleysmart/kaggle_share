import os
import numpy as np

import keras 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation, Flatten

# metrics
import libs.metrics
import libs.timing

from keras.callbacks import Callback
import sklearn.metrics as sklm

class RoiModel:
    # this model has some interasting points to note
    #  - its a cnn model
    #  - it trains using a mask data generator that outputs patches and expected masks
    #     but it takes the masks and reduces them to a true/false signal
    #  - it predicts if the region has something interesting for the  class

    def setup_model(self):
        # ok model death ..  ModelTestBed.ipynb
        inputs = Input((self.patch_size, self.patch_size, self.in_chan))


        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool  = MaxPooling2D(pool_size=(2, 2))(conv1)

        if self.patch_size > 20:
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
            pool  = MaxPooling2D(pool_size=(2, 2))(conv2)

        if self.patch_size > 40:
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
            pool  = MaxPooling2D(pool_size=(2, 2))(conv3)

        if self.patch_size > 80:
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
            pool  = MaxPooling2D(pool_size=(2, 2))(conv4)

        if self.patch_size > 160:
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool)
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
            pool  = MaxPooling2D(pool_size=(2, 2))(conv5)

        dense = Flatten()(pool)

        if self.patch_size > 80:
           dense = Dense(256, activation='relu')(dense)
        if self.patch_size > 40:
            dense = Dense(128, activation='relu')(dense)
        if self.patch_size > 20:
            dense = Dense(64, activation='relu')(dense)
        if self.patch_size > 10:
            dense = Dense(16, activation='relu')(dense)

        output = Dense(1,activation='sigmoid')(dense)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(),
            loss='binary_crossentropy', 
            metrics=['accuracy',
                libs.metrics.f1_score, 
                libs.metrics.precision, 
                libs.metrics.recall])

        print model.summary()

        return model

    def load(self):
        best_file = self.weights_file + "_best.hdf5"
        if os.path.isfile(best_file):
            print "loading prior weights from:", best_file
            self.model.load_weights(best_file)

        # TODO save and load threasholds
        self.threasholds = [0.5]

    def train(self, generator,
        fit_epochs=1, fit_batch=4,
        val_samples=5000,
        gen_epochs=1, gen_samples=10000):

        mark = libs.timing.sync()
        print "training net", self.weights_file

        # TODO precache the validation sets

        # build some vaidation data
        # note yes i know this is poluting my validation pool with training pool
        mark = libs.timing.timing(mark)
        print "creating validation set.."
        valid_in  = None
        valid_out = None
        for patches_in, roi_out in generator(epochs=1, samples=val_samples):
            if valid_in is None:
                valid_in  = patches_in .copy()
                valid_out = roi_out.copy()
            else:
                valid_in  = np.concatenate((valid_in,  patches_in), axis=0)
                valid_out = np.concatenate((valid_out, roi_out   ), axis=0)

        class_pos = np.sum(valid_out) 
        print "val class mix:", class_pos, "/", valid_out.shape[0], "(", (class_pos/valid_out.shape[0]), ")"

        model_checkpoint = ModelCheckpoint(self.weights_file + "_best.hdf5", 
            monitor='loss', 
            save_best_only=True)

        mark = libs.timing.timing(mark)
        print "training cycle....."
        for patches_in, roi_out in generator(epochs=gen_epochs, samples=gen_samples):

            mark = libs.timing.timing(mark)

            class_pos = np.sum(roi_out)
            print "train class mix:", class_pos, "/", roi_out.shape[0], " (", (class_pos/roi_out.shape[0]), ")"
            print "starting training..... "
            metrics = libs.metrics.Metrics()

            self.model.fit(patches_in, roi_out, 
                    batch_size=fit_batch, 
                    epochs=fit_epochs, verbose=1, shuffle=True,
                    callbacks=[model_checkpoint, metrics],
                    validation_data=(valid_in, valid_out))

            print "final.. "
            print " confusion:"
            print metrics.confusion
            print " precision:", metrics.precision
            print " recall   :", metrics.recall
            print " f1s      :", metrics.f1s
            print " kappa    :", metrics.kappa
            print " auc      :", metrics.auc

            mark = libs.timing.timing(mark)
            print "done training........"

            mark = libs.timing.timing(mark)
            print "next training step...."

        print "done training"

    def predict(self,patches):
        prediction = self.model.predict(patches, batch_size=8) 
        return prediction

    # note we expect a single image here
    def threshold(self, img):
        for i in range(img.shape[2]):
            img[:,:,i] = img[:,:,i] > self.threasholds[i]

    def __init__(self,file,patch_size,in_chan,out_chan):
        self.weights_file = file
        self.patch_size   = patch_size
        self.in_chan      = in_chan
        self.out_chan     = out_chan
        self.model = self.setup_model()
