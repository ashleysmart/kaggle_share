# note this fits into the patch_generator.py

from keras.models import Model
import numpy as np
import tensorflow as tf
import cv2

class BottlenecksTransform:
    # ACS TODO rewrite.. enum vs policy.. policy wins due to open close principle
    # todo add the other models https://keras.io/applications/

    VGG16           = 1
    VGG19           = 2
    RESNET50        = 3
    INCEPTIONV3     = 4
    XCEPTION        = 5
    INCEPTIONRES2V3 = 6
    MOBILENET       = 7
    
    def __init__(self, model_type=INCEPTIONV3,  weights="imagenet"):       
        if model_type == BottlenecksTransform.VGG16:
            import keras.applications.vgg16

            base_model   = keras.applications.vgg16.VGG16(weights=weights)
            output_layer = base_model.get_layer('fc1')
            
            self.preprocessor = keras.applications.vgg16.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
	    self.input_size  = (224, 224)
	    self.output_size = output_layer.output_shape
        elif model_type == BottlenecksTransform.VGG19:
            import keras.applications.vgg19

            base_model = keras.applications.vgg19.VGG19(weights=weights)
            output_layer = base_model.get_layer('fc1')

            self.preprocessor = keras.applications.vgg19.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
            self.input_size = (224, 224)
	    self.output_size = output_layer.output_shape
        elif model_type == BottlenecksTransform.RESNET50:
            import keras.applications.resnet50

            base_model = keras.applications.resnet50.ResNet50(weights=weights)
            output_layer = base_model.get_layer('flatten')

            self.preprocessor = keras.applications.resnet50.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
            self.input_size = (224, 224)
	    self.output_size = output_layer.output_shape
        elif model_type == BottlenecksTransform.INCEPTIONV3:
            import keras.applications.inception_v3

            base_model = keras.applications.inception_v3.InceptionV3(weights=weights)
            output_layer = base_model.get_layer('avg_pool')

            self.preprocessor = keras.applications.inception_v3.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
            self.input_size = (299, 299)
            self.output_size = output_layer.output_shape
        elif model_type == BottlenecksTransform.XCEPTION:
            import keras.applications.xceptione

            base_model = keras.applications.xception.Xception(weights=weights)
            output_layer = base_model.get_layer('avg_pool')

            self.preprocessor = keras.applications.xception.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
            self.input_size = (299, 299)
	    self.output_size = output_layer.output_shape

        elif model_type == BottlenecksTransform.MOBILENET:
            import keras.applications.mobilenet

            base_model = keras.applications.mobilenet.MobileNet(weights=weights)
            output_layer = base_model.get_layer('act_softmax')

            self.preprocessor = keras.applications.mobilenet.preprocess_input
	    self.model = Model(inputs=base_model.input, outputs=output_layer.output)
            self.input_size = (224, 224)
	    self.output_size = output_layer.output_shape
        else:
            raise ValueError("no idea what " + model_name + " is..")

    # allocate for multiple ouputs from __call__
    def allocate(self,samples,patch_size,channels):
        shape = (samples,) + self.output_size[1:]
        return np.zeros(shape)

    # designed for single data point
    def __call__(self, data):
        # data shape: height, width, channel

        # ouch my gpu doesnt have the memory for these large  models..
        # make certain we are using the cpu..
        with tf.device('/cpu:0'):
            # resize patch into networks input size
            scaled = np.zeros(self.input_size + (data.shape[2],))
            for i in range(data.shape[2]):
                scaled[:,:,i] = cv2.resize(data[:,:,i],self.input_size, interpolation = cv2.INTER_CUBIC)
                
            # the preprocessing is model specifc.. (hope this doesnt compound oddly)
            pre_processed = self.preprocessor(scaled)

            # generate the bottlenext features
            pre_processed = pre_processed.reshape((1,) + pre_processed.shape)
            features = self.model.predict(pre_processed)

            features = features.reshape(features.shape[1:])
                   
        return features

                
