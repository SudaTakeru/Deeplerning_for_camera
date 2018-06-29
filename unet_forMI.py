# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:44:42 2018

@author: Takeru_2
"""

from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers import Lambda
from keras.regularizers import l1, l2, Regularizer
from keras.constraints import non_neg
import numpy as np
from keras import backend as K

class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count,S,psize):
        self.INPUT_IMAGE_SIZE = psize
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.bandwidth = 3
        self.Wl2=1E-4*3/31
        ginter=np.array([[0,0.25,0],[0.25,1,0.25],[0,0.25,0]])
        rinter=np.array([[0.25,0.5,0.25],[0.5,1,0.5],[0.25,0.5,0.25]])
        binter=np.array([[0.25,0.5,0.25],[0.5,1,0.5],[0.25,0.5,0.25]])

        conw = np.ones((1,1,3,1))
        conwb = np.zeros((1,))
        ipw = np.zeros((3,3,3,3))
        ipwb = np.zeros((3,))
        ipw[:,:,2,2]=rinter
        ipw[:,:,1,1]=ginter
        ipw[:,:,0,0]=binter
        ##color filter
        RF=np.arange(psize*psize).reshape((psize,psize))
        Rfarray = (RF//psize)%2 & RF%2 == 0
        nRfarray = (RF//psize)%2 & RF%2 != 0
        RF[Rfarray] = 0
        RF[nRfarray] = 1

        BF=np.arange(psize*psize).reshape((psize,psize))
        Bfarray = (BF//psize)%2-1 & BF%2-1 == 0
        nBfarray = (BF//psize)%2-1 & BF%2-1 != 0
        BF[Bfarray] = 0
        BF[nBfarray] = 1
  
        GF=BF+RF
        Gfarray = GF == 1
        nGfarray = GF == 0
        GF[Gfarray] = 0
        GF[nGfarray] = 1

        CF0=[]
        CF0.append(RF)
        CF0.append(GF)
        CF0.append(BF)
        CF0=np.transpose(np.array(CF0))
        CF=np.zeros((1,psize,psize,3))
        CF[0]=CF0
        constants = CF
        mask = K.variable(constants)

        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        
        # カメラ感度
        rgbimg = Conv2D( self.bandwidth, (1,1), padding='same', activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l1(self.Wl2),kernel_constraint=non_neg(), use_bias=False, weights=[S],name='Sensitivitiy')(inputs) 

        mosaic = Lambda(lambda x: x*mask)(rgbimg)
        mosaic1= Conv2D(1,(1,1), padding='same', activation='relu', use_bias=True, weights=[conw, conwb],trainable = False)(mosaic)
        interpolated = Conv2D(3,(3,3), padding='same', activation='relu',use_bias=True, weights=[ipw, ipwb],trainable = False)(mosaic)
        mosaics = Concatenate()([mosaic,mosaic1,interpolated])



        # エンコーダーの作成
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(mosaics)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)
        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, enc8)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = first_layer_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(input=inputs, output=dec8)
        self.rgbimgmodel = Model(input=inputs,output=rgbimg)
        
    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET
    
    def get_submodel(self):
        return self.rgbimgmodel
    
    
    