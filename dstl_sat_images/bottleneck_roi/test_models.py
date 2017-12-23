import keras 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten

import libs.metrics

def models():
    return { "test_model1": setup_model_1,
        "test_model2": setup_model_2,
        "test_model3": setup_model_3 
         }

def setup_model_1(size):
    # Todo - add some drop out..
    inputs = Input((size,))
    output = Dense(10,activation='sigmoid')(inputs)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model

def setup_model_2(size):
    # Todo - add some drop out..
    inputs = Input((size,))
    
    x      = Dense(512,activation='relu')(inputs)
    x      = Dense(32,activation='relu')(x)
    output = Dense(10,activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model


def setup_model_3(size):
    # Todo - add some drop out..
    inputs = Input((size,))
    
    x      = inputs
    x      = Dense(1024,activation='relu')(x)
    x      = Dense(512,activation='relu')(x)
    x      = Dense(256,activation='relu')(x)
    x      = Dense(128,activation='relu')(x)
    output = Dense(10,activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=['accuracy',
            libs.metrics.f1_score, 
            libs.metrics.precision, 
            libs.metrics.recall])

    print model.summary()

    return model
