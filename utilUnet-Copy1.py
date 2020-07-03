import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add, multiply
from tensorflow.keras.layers import Lambda, UpSampling2D, Cropping2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
import tensorflow
from numpy.random import seed
from tensorflow.keras.preprocessing import image
import math
import json
import random

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    #print(x.shape)
    #print(x)
    if batchnorm:
        x = BatchNormalization()(x)
        #print(x.shape)
        #print(x)
    x = Activation('relu')(x)
    #print(x.shape)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    #print(x.shape)
    if batchnorm:
        x = BatchNormalization()(x)
        #print(x.shape)
    x = Activation('relu')(x)
    #print(x.shape)
    
    return x

def conv2d_single(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet2cl(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    print("c1= "+str(c1.shape))
    p1 = MaxPooling2D((2, 2))(c1)
    #print(p1.shape)
    p1 = Dropout(dropout)(p1)
    #print(p1.shape)
    #print(c1.shape)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    print("c2= "+str(c2.shape))
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    print("c3= "+str(c3.shape))
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    print("c4= "+str(c4.shape))
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    print("c5= "+str(c5.shape))
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    print("u6= "+str(u6.shape))
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    print("c6= "+str(c6.shape))
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    print("u7= "+str(u7.shape))
    u7 = concatenate([u7, c3])
    print("u7 concatenate = "+str(u7.shape))
    u7 = Dropout(dropout)(u7)
    print("u7 dropout = "+str(u7.shape))
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    print("c7= "+str(c7.shape))
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    print("u8= "+str(u8.shape))
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    print("c8= "+str(c8.shape))
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    print("u9= "+str(u9.shape))
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    print("c9= "+str(c9.shape))
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_unet1cl(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_single(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    #print(c1.shape)
    
    c2 = conv2d_single(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    #print(c2.shape)
    
    c3 = conv2d_single(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    #print(c3.shape)
    
    c4 = conv2d_single(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    #print(c4.shape)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_single(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    print(c5.shape)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_single(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_single(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_single(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_single(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def UnetGatingSignal(input_tensor, batchnorm=False):
    shape = K.int_shape(input_tensor)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape):
        # inter_shape = 256 (channels)
        shape_x = K.int_shape(x)  # 32x32 128
        shape_g = K.int_shape(g)  # 16x16 512

        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16x16 256
        shape_theta_x = K.int_shape(theta_x) # 16x16 256

        phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)  # 16x16 256
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], 
                                                                  shape_theta_x[2] // shape_g[2]),
                                     padding='same')(phi_g) # 16x16 256 stride is (1x1)

        concat_xg = add([upsample_g, theta_x]) # 16x16 256
        act_xg = Activation('relu')(concat_xg) # 16x16 256
        psi = Conv2D(1, (1, 1), padding='same')(act_xg) # 16x16 1
        sigmoid_xg = Activation('sigmoid')(psi) # 16x16 1
        shape_sigmoid = K.int_shape(sigmoid_xg) # 16x16 1
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], 
                                          shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32x32 1


        upsample_psi = expend_as(upsample_psi, shape_x[3]) # 32x32 128

        y = multiply([upsample_psi, x]) # 32x32 128

        result = Conv2D(shape_x[3], (1, 1), padding='same')(y) # 32x32 128
        result_bn = BatchNormalization()(result) # 32x32 128
        return result_bn

def get_attnunet1cl(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # input_img (256x256 1)
    """Function to define the UNET Model"""
    
    # Contracting Path
    c1 = conv2d_single(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    # c1 (256x256 16)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    # p1 (128x128 16)
    
    c2 = conv2d_single(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    # c2 (128x128 32)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    # p2 (64x64 32)
    
    c3 = conv2d_single(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    # c3 (64x64 64)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    # p3 (32x32 64)
    
    c4 = conv2d_single(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    # c4 (32x32 128)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    # p4 (16x16 128)
    
    
    c5 = conv2d_single(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    # c5 (16x16 256)
    
    # Expansive Path
    
    # Create a gating signal out of c5
    gating = UnetGatingSignal(c5, batchnorm = batchnorm)
    # g (16x16 512)
    # Creating attention gate out of c4 and gating
    inter_shape = 2*(K.int_shape(c4)[3])
    attn_c4g = AttnGatingBlock(c4, gating, inter_shape) #32x32 128 (will match shape of c4)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5) #32x32 128
    u6 = concatenate([u6, attn_c4g],axis=3) #32x32 256
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_single(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm) #32x32 128
    
    # Create a gating signal out of c6
    gating = UnetGatingSignal(c6, batchnorm = batchnorm)
    # g (32x32 256)
    # Creating attention gate out of c3 and gating
    inter_shape = 2*(K.int_shape(c3)[3])
    attn_c3g = AttnGatingBlock(c3, gating, inter_shape) #64x64 64 (will match shape of c3)
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6) #64x64 64
    u7 = concatenate([u7, attn_c3g],axis=3) #64x64 128
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_single(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm) #64x64 64
    
    # Create a gating signal out of c7
    gating = UnetGatingSignal(c7, batchnorm = batchnorm)
    # g (64x64 128)
    # Creating attention gate out of c2 and gating
    inter_shape = 2*(K.int_shape(c2)[3])
    attn_c2g = AttnGatingBlock(c2, gating, inter_shape) #128x128 32 (will match shape of c2)
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7) #128x128 32
    u8 = concatenate([u8, attn_c2g],axis=3) #128x128 64
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_single(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm) #128x128 32
    
    # Create a gating signal out of c8
    gating = UnetGatingSignal(c8, batchnorm = batchnorm)
    # g (128x128 64)
    # Creating attention gate out of c1 and gating
    inter_shape = 2*(K.int_shape(c4)[3])
    attn_c1g = AttnGatingBlock(c1, gating, inter_shape) #256x256 16 (will match shape of c1)
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8) #256x256 16
    u9 = concatenate([u9, attn_c1g],axis=3) #256x256 16
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_single(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm) #256x256 16
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9) #256x256 1
    model = Model(inputs=[input_img], outputs=[outputs])
    return model