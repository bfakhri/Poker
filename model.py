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
        num_class_lyrs = 2 

        # Multi-scale feature extractor layers
        self.fe_lyrs = []
        for i in range(num_conv_lyrs):
            self.fe_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        # Classification layers
        self.class_lyrs = []
        for i in range(num_class_lyrs):
            if(i == num_class_lyrs-1):
                act = None
            else: 
                act = 'relu'
            self.class_lyrs.append(tf.keras.layers.Dense(units=num_filters, activation=act))

    def call(self, x, reconstruct=True):
        ''' 
        Trains as an autoencoder but provides features during inference
        '''
        h = x
        # Small scale feature extractor layers
        for idx,layer in enumerate(self.fe_lyrs):
            h = layer(h)

        # Reduce on spatial dimensions (bs, w, h, c) -> (bs, c)
        (bs, width, height, c) = h.shape
        #h = tf.nn.max_pool(h, [1,w, h,1], [1,1,1,1], padding='SAME')
        h = tf.math.reduce_max(h, axis=1)
        h = tf.math.reduce_max(h, axis=1)
        for idx,layer in enumerate(self.class_lyrs):
            h = layer(h)

        preds = tf.nn.sigmoid(h)

        return preds

        
