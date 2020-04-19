#################################################################
##################################################################

import tensorflow as tf
import numpy as np

class CardClassifier(tf.keras.Model):
    ''' 
    Input/Output: (bs, h, w, c) -> (bs, 52) 
    '''
    def __init__(self):
        super(CardClassifier, self).__init__()

        # Params
        num_filters = 52
        num_conv_lyrs = 16

        # Multi-scale feature extractor layers
        self.fe_lyrs = []
        for i in range(num_conv_layers):
            self.fe_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        # Classification layers
        self.class_lyrs = []
        for i in range(num_conv_layers):
            self.class_lyrs.append(tf.keras.layers.Dense(units=num_filters, activation='relu'))

    def call(self, x, reconstruct=True):
        ''' 
        Trains as an autoencoder but provides features during inference
        '''
        h = x
        print('Orig: ', h.shape)
        # Small scale feature extractor layers
        for idx,layer in enumerate(self.fe_lyrs):
            h = layer(h)
            print(idx, h.shape)

        # Reduce on spatial dimensions (bs, w, h, c) -> (bs, c)
        (bs, w, h, c) = h.shape
        h = tf.nn.max_pool(h, [w, h])
        for idx,layer in enumerate(self.class_lyrs):
            h = layer(h)
            print(idx, h.shape)

        preds = tf.nn.sigmoid(h)

        return preds

        


        
class SOD_Model:
    ''' 
    Object detector
    '''
    def __init__(self):
        super(SOD_Model, self).__init__()

        # Model Params
        num_filters = 8
        output_size = 10

        # Simple object detection layers
        self.od_lyrs = []
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        self.od_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', activation='relu'))

