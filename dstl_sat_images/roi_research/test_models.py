import keras 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.constraints import maxnorm

import libs.metrics

def models():
    return { "test_model1": setup_model_1,
        "test_model2": setup_model_2,
        "test_model3": setup_model_3,
        "test_model3b": setup_model_3b,
        "test_model4": setup_model_4,
        "test_model5": setup_model_5 }

def setup_model_1(patch_size, in_chan):
    # Todo - add some drop out..
    inputs = Input((patch_size, patch_size, in_chan))
    flat = Flatten()(inputs)     
    output = Dense(1,activation='sigmoid')(flat)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model


def setup_model_2(patch_size, in_chan):
    # Todo - add some drop out..
    inputs = Input((patch_size, patch_size, in_chan))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    flat = Flatten()(pool1)     
    output = Dense(1,activation='sigmoid')(flat)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model

def setup_model_3(patch_size, in_chan):
    # Todo - add some drop out..
    inputs = Input((patch_size, patch_size, in_chan))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = Flatten()(pool3)     
    dense8 = Dense(64, activation='relu')(flat)
    dense9 = Dense(16, activation='relu')(dense8)
    output = Dense(1,activation='sigmoid')(dense9)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model

def setup_model_3b(patch_size, in_chan):
    inputs = Input((patch_size, patch_size, in_chan))
    drop1 = Dropout(0.1)(inputs)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(drop1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    drop2 = Dropout(0.05)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(drop2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    drop3 = Dropout(0.05)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(drop3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = Flatten()(pool3)     
    dense8 = Dense(64, activation='relu', kernel_constraint=maxnorm(3))(flat)
    dense9 = Dense(16, activation='relu', kernel_constraint=maxnorm(3))(dense8)
    output = Dense(1,activation='sigmoid')(dense9)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(decay=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy',
                           libs.metrics.f1_score, 
                           libs.metrics.precision, 
                           libs.metrics.recall])

    print model.summary()

    return model



def setup_model_4(patch_size, in_chan):
    # Todo - add some drop out..
    inputs = Input((patch_size, patch_size, in_chan))

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

    flat = Flatten()(pool4)

    dense6 = Dense(512, activation='relu')(flat)
    dense7 = Dense(128, activation='relu')(dense6)
    dense8 = Dense(64, activation='relu')(dense7)
    dense9 = Dense(16, activation='relu')(dense8)
    output = Dense(1,activation='sigmoid')(dense9)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model

def setup_model_5(patch_size, in_chan):
    # Todo - add some drop out..
    inputs = Input((patch_size, patch_size, in_chan))

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

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    flat = Flatten()(pool5)

    dense6 = Dense(512, activation='relu')(flat)
    dense7 = Dense(128, activation='relu')(dense6)
    dense8 = Dense(64, activation='relu')(dense7)
    dense9 = Dense(16, activation='relu')(dense8)
    output = Dense(1,activation='sigmoid')(dense9)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model

def setup_model_actual(patch_size, in_chan):
    # ok model death ..  ModelTestBed.ipynb
    inputs = Input((patch_size, patch_size, in_chan))


    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool  = MaxPooling2D(pool_size=(2, 2))(conv1)

    if patch_size > 20:
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool  = MaxPooling2D(pool_size=(2, 2))(conv2)

    if patch_size > 40:
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool  = MaxPooling2D(pool_size=(2, 2))(conv3)

    if patch_size > 80:
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool  = MaxPooling2D(pool_size=(2, 2))(conv4)

    if patch_size > 160:
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        pool  = MaxPooling2D(pool_size=(2, 2))(conv5)

    dense = Flatten()(pool)

    if patch_size > 160:
       dense = Dense(512, activation='relu')(dense)
    if patch_size > 80:
        dense = Dense(128, activation='relu')(dense)
    if patch_size > 40:
        dense = Dense(64, activation='relu')(dense)
    if patch_size > 20:
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
