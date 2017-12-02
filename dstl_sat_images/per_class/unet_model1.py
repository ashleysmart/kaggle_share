import os
import numpy as np

import keras 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# metrics
import libs.metrics
import libs.timing

from sklearn.metrics import jaccard_similarity_score

class UnetModel:
    # this model has some interasting points to note
    #  - its a unet model
    #  - it trains using a data generator that outputs patches and expected masks
    #  - it predicts output masks when given a patch 

    def calc_thresholds(self, patches_in, patches_out):
        prediction = self.model.predict(patches_in, batch_size=4)
        avg, trs = [], []

        for i in range(self.out_chan):
            t_prd = prediction [:, :, :, i]
            t_msk = patches_out[:, :, :, i]

            t_prd = t_prd.reshape(t_msk.shape[0] * t_msk.shape[1], t_msk.shape[2])
            t_msk = t_msk.reshape(t_msk.shape[0] * t_msk.shape[1], t_msk.shape[2])

            t_msk = t_msk > 0.5 
            # threshold finder
            best_score = 0
            best_threashold = 0
            for j in range(10):
                threashold = (j+1) / 10.0
                threshold_mask = (t_prd > threashold) 

                jk = jaccard_similarity_score(t_msk, threshold_mask)
                if jk > best_score:
                    best_score = jk
                    best_threashold = threashold
            print " -- output:", i, "best:", best_score, "threashold:", best_threashold
            avg.append(best_score)
            trs.append(best_threashold)

        score = sum(avg) / 10.0
        return score, trs


    def setup_model(self):
        # TODO drop out?
        inputs = Input((self.patch_size, self.patch_size, self.in_chan))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        up6    = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([up6, conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        up8 = UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(self.out_chan, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(),
            loss='binary_crossentropy', 
            metrics=[libs.metrics.jaccard_coef, 
                    libs.metrics.jaccard_coef_int, 
                    'accuracy'])

        #print model.summary()

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
        val_samples=3000,
        gen_epochs=1, gen_samples=10000):

        mark = datetime.datetime.now()
        print "training net", self.weights_file

        # TODO precache the validation sets

        # build some vaidation data
        # note yes i know this is poluting my validation pool with training pool
        mark = libs.timing.timing(mark)
        print "creating validation set.."
        valid_in  = None
        valid_out = None
        for patches_in, patches_out in generator(epochs=1, samples=val_samples):
            if valid_in is None:
                valid_in  = patches_in .copy()
                valid_out = patches_out.copy()
            else:
                valid_in  = np.concatenate((valid_in,  patches_in ), axis=0)
                valid_out = np.concatenate((valid_out, patches_out), axis=0)

        model_checkpoint = ModelCheckpoint(self.weights_file + "_best.hdf5", 
            monitor='loss', 
            save_best_only=True)

        mark = libs.timing.timing(mark)
        print "training cycle....."
        for patches_in, patches_out in generator(epochs=gen_epochs, samples=gen_samples):
            mark = libs.timing.timing(mark)
            print "starting training..... "
            self.model.fit(patches_in, patches_out, 
                    batch_size=fit_batch, 
                    epochs=fit_epochs, verbose=1, shuffle=True,
                    callbacks=[model_checkpoint], 
                    validation_data=(valid_in, valid_out))

            mark = libs.timing.timing(mark)
            print "done training........"

            score, trs = self.calc_thresholds(valid_in,valid_out)
            self.thresholds = trs

            print 'validation jacc:', score
            np.save(self.weights_file + ('_thresholds_%.4f' % score),  self.thresholds)
            self.model.save_weights(self.weights_file + ('_jacc_%.4f.hdf5' % score))

            mark = libs.timing.timing(mark)
            print "next training step...."

    def predict(self,patches):
        # TODO.. well code it... 
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
