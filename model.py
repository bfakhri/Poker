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
        num_layers = 16

        # Multi-scale feature extractor layers
        self.fe_lyrs = []
        for i in range(num_layers):
            self.fe_lyrs.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        # Classification layers
        self.recon_lyr = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')

    def call(self, x, reconstruct=True):
        ''' 
        Trains as an autoencoder but provides features during inference
        '''
        h = x
        saved_h = []
        print('Orig: ', h.shape)
        # Small scale feature extractor layers
        for idx,layer in enumerate(self.fe_lyrs):
            h = layer(h)
            if(np.any(idx == self.stops)):
                saved_h.append(h)
                print(idx, h.shape, '\tSaved!')
            else:
                print(idx, h.shape)

        if(reconstruct):
            # Concat all the saved features for reconstruction
            saved_h_rs = []
            for feats in saved_h:
                print('Resizing: ', feats.shape[1:3], ' to: ', x.shape[1:3])
                saved_h_rs.append(tf.image.resize(feats, x.shape[1:3]))
                print('new size: ', saved_h_rs[-1].shape)
            h = tf.concat(saved_h_rs, axis=-1)
            print('Concated: ', h.shape)
            # Perform reconstruction
            h = self.recon_lyr(h)
            print('Reconstructed: ', h.shape)
            return h 
        else:
            return saved_h


        
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

