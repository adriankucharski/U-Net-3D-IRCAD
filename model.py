import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, softmax, tanh
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,Conv2D,
                                     Dropout, Input, LeakyReLU, MaxPooling3D,MaxPooling2D,
                                     Softmax, UpSampling3D, UpSampling2D, concatenate)
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#################################################
def tversky_loss(alfa = 0.2, beta=0.8):
  def loss(y_true, y_pred):
    numerator = y_true * y_pred
    denominator = y_true * y_pred + (alfa * (1 - y_true) * y_pred) + ((1 - beta) * y_true * (1 - y_pred))

    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

  return loss

def iou_loss():
    def loss(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(K.abs(y_true + y_pred))
        return 1 - 2 * intersection / union

    return loss

def my_loss(smooth = 1e-4):
    def loss(y_true, y_pred):
        ratio_tp = K.sum(y_true)
        ratio_tn = K.sum(1 - y_true)

        possion = K.mean(y_pred - y_true * K.log(y_pred))
        test = K.mean(y_pred - (1 - y_true) * K.log(y_pred)) * (ratio_tp/ratio_tn)


        return 1 - possion / test

    return loss

def dice_coef(y_true, y_pred, smooth=1e-4):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

###############################################
def SegNet3DBlock(inputs, layers, filters, drop = 0.2, kernel_size=(3,3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'):
    conv = inputs
    for _ in range(layers):
        conv = Conv3D(filters, kernel_size, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
    conv = Dropout(drop)(conv)
    return conv

def SegNet3D(shape, weights = None):
    inputs = Input(shape)
    conv, pool = inputs, inputs

    #encoder
    for numOfFilters in [4, 8, 16, 32]:
        conv = SegNet3DBlock(pool, layers=2, filters=numOfFilters)
        pool = MaxPooling3D((2,2,2))(conv)

    conv = SegNet3DBlock(pool, layers=3, filters=128)

    #decoder
    for numOfFilters in [64, 32, 16, 8]:
        upsam = UpSampling3D((2,2,2))(conv)
        conv  = SegNet3DBlock(upsam, layers=2, filters=numOfFilters)
    
    conv = SegNet3DBlock(upsam, layers=2, filters=4)

    outputs = Conv3D(1, 1, activation='sigmoid')(conv)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics = [Precision(), Recall(), AUC(), Accuracy()])
    model.summary()

    return model

def UNet3DBlock(inputs, layers, filters, kernel_size=(3,3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dropout = 0.25):
    conv = inputs
    for _ in range(layers):
        conv = Conv3D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv)
    conv = Dropout(dropout)(conv)
    return conv

def UNet3D(shape, weights = None):
    inputs = Input(shape)

    #encoder
    x = inputs
    conv_encoder = []
    for filters in [4, 8, 16, 32]:
        conv = UNet3DBlock(x, layers=2, filters=filters)
        x = MaxPooling3D(pool_size=(2,2,2))(conv)
        conv_encoder.append(conv)

    x = UNet3DBlock(x, layers=2, filters=128)

    #decoder
    for filters in [64, 32, 16, 8]:
        x = UNet3DBlock(x, layers=2, filters=filters)
        x = UpSampling3D(size=(2,2,2))(x)
        x = concatenate([conv_encoder.pop(), x])

    outputs = Conv3D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'binary_crossentropy', metrics = [AUC(), Accuracy()])
    model.summary()

    return model


#################################################
def UNet2DBlock(inputs, layers, filters, kernel_size=(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', dropout = 0.2):
    conv = inputs
    for _ in range(layers):
        conv = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv)
    conv = Dropout(dropout)(conv)
    return conv

def UNet2D(shape, weights = None):
    inputs = Input(shape)

    #encoder
    x = inputs
    conv_encoder = []
    for filters in [16, 32, 64, 128, 128]:
        conv = UNet2DBlock(x, layers=2, filters=filters, dropout=0.2)
        x = MaxPooling2D(pool_size=(2,2))(conv)
        conv_encoder.append(conv)

    x = UNet2DBlock(x, layers=2, filters=512, dropout=0.5)

    #decoder
    for filters in [256, 256, 128, 64, 32]:
        x = UNet2DBlock(x, layers=2, filters=filters, dropout=0.2)
        x = UpSampling2D(size=(2,2))(x)
        x = concatenate([conv_encoder.pop(), x])
    
    x = UNet2DBlock(x, layers=2, filters=32, dropout=0.0)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'binary_crossentropy', metrics = [AUC(), Accuracy()])
    model.summary()
    if weights is not None:
        model.load_weights(weights)
    return model
