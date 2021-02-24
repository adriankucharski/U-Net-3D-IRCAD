import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D, Conv2D,
                                     Dropout, Input, LeakyReLU, MaxPooling3D, MaxPooling2D,
                                     Softmax, UpSampling3D, UpSampling2D, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, MeanIoU
from tensorflow.keras.activations import sigmoid, softmax, tanh
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

#################################################


def tversky_loss(alfa=0.2, beta=0.8):
    def loss(y_true, y_pred):
        numerator = y_true * y_pred
        denominator = y_true * y_pred + \
            (alfa * (1 - y_true) * y_pred) + \
            ((1 - beta) * y_true * (1 - y_pred))

        return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

    return loss


def iou_loss():
    def loss(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(K.abs(y_true + y_pred))
        return 1 - 2 * intersection / union

    return loss


def my_loss(smooth=1e-4):
    def loss_1(true, pred, s = smooth):
        inter = K.sum(true * K.square(pred)) + s
        sums = K.sum(true) + s
        return 1 - inter/sums 

    return loss_1


def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection =  2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth/2.0
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return intersection / sums


def focal_sum_loss(alpha=0.25, gamma=2.0, smooth = 0.0):
    def loss(y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy(alpha = alpha, gamma=gamma)
        s = fl(y_true, y_pred)
        return K.mean(s) 
    return loss

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

###############################################


def SegNet3DBlock(inputs, layers, filters, drop=0.2, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', strides=1):
    conv = inputs
    for _ in range(layers):
        conv = Conv3D(filters, kernel_size, activation='linear',
                      padding=padding, kernel_initializer=kernel_initializer, strides=strides)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
    conv = Dropout(drop)(conv)
    return conv


def SegNet3D(shape, weights=None):
    inputs = Input(shape)
    conv, pool = inputs, inputs

    # encoder
    for numOfFilters in [4, 8, 16, 32]:
        conv = SegNet3DBlock(pool, layers=2, filters=numOfFilters)
        pool = MaxPooling3D((2, 2, 2))(conv)

    conv = SegNet3DBlock(pool, layers=3, filters=128)

    # decoder
    for numOfFilters in [64, 32, 16, 8]:
        upsam = UpSampling3D((2, 2, 2))(conv)
        conv = SegNet3DBlock(upsam, layers=2, filters=numOfFilters)

    conv = SegNet3DBlock(upsam, layers=2, filters=4)

    outputs = Conv3D(1, 1, activation='sigmoid')(conv)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy',
                  metrics=[Precision(), Recall(), AUC(), Accuracy()])
    model.summary()

    return model

###############################################

def UNet3DBlock(inputs, layers, filters, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', dropout=0.25, strides=1):
    conv = inputs
    for _ in range(layers):
        conv = Conv3D(filters, kernel_size, activation=activation,
                      padding=padding, kernel_initializer=kernel_initializer, strides=strides)(conv)
    conv = Dropout(dropout)(conv)
    return conv


def UNet3D(shape, weights=None):
    conv_encoder = []
    encoder_filters = np.array([4, 8, 16, 32])
    decoder_filters = np.array([64, 32, 16, 8])
    bridge_filters = 128

    inputs = Input(shape)
    # encoder
    x = inputs
    for filters in encoder_filters:
        conv = UNet3DBlock(x, layers=2, filters=filters)
        x = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet3DBlock(x, layers=2, filters=bridge_filters)

    # decoder
    for filters in decoder_filters:
        x = UNet3DBlock(x, layers=2, filters=filters)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet3DBlock(x, layers=2, filters=8)
    outputs = Conv3D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=[AUC(), dice_coef])
    model.summary()

    return model

def UNet3DPatch(shape, weights=None):
    conv_encoder = []
    encoder_filters = np.array([2, 4, 8, 16])
    decoder_filters = np.array([32, 16, 8, 4])
    bridge_filters = 64

    inputs = Input(shape)
    # encoder
    x = inputs
    for filters in encoder_filters:
        conv = UNet3DBlock(x, layers=2, filters=filters)
        x = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet3DBlock(x, layers=2, filters=bridge_filters)

    # decoder
    for filters in decoder_filters:
        x = UNet3DBlock(x, layers=2, filters=filters)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet3DBlock(x, layers=2, filters=8)
    outputs = Conv3D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=[AUC(), dice_coef])
    model.summary()

    return model


#################################################

def UNet2DBlock(inputs, layers, filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', dropout=0.2):
    conv = inputs
    for _ in range(layers):
        conv = Conv2D(filters, kernel_size, activation=activation, bias_initializer='random_normal',
                      padding=padding, kernel_initializer=kernel_initializer, use_bias=True)(conv)
    conv = Dropout(dropout)(conv)
    return conv

def UNet2D(shape, weights=None):
    conv_encoder = []
    encoder_filters = np.array([16, 32, 64, 128, 256])
    decoder_filters = np.array([256, 256, 128, 64, 32])
    bridge_filters = 512

    inputs = Input(shape)
    # encoder
    x = inputs
    for filters in encoder_filters:
        conv = UNet2DBlock(x, layers=2, filters=filters, dropout=0.5)
        x = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet2DBlock(x, layers=2, filters=bridge_filters, dropout=0.5)

    # decoder
    for filters in decoder_filters:
        x = UNet2DBlock(x, layers=2, filters=filters, dropout=0.5)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet2DBlock(x, layers=2, filters=32, dropout=0.5)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'binary_crossentropy', metrics = [AUC(), dice_coef])

    model.summary()
    if weights is not None:
        model.load_weights(weights)
    return model
#################################################

def experimental_network(shape, weights=None):
    inputs = Input(shape)

    # encoder
    x = inputs
    conv_encoder = []
    for filters in [16, 32, 64, 128, 256]:
        conv = UNet2DBlock(x, layers=2, filters=filters, dropout=0.4)
        x = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet2DBlock(x, layers=2, filters=256, dropout=0.5)

    # decoder
    for filters in [256, 128, 64, 32, 16]:
        x = UNet2DBlock(x, layers=2, filters=filters, dropout=0.4)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = Conv2D(1, 1, activation='tanh')(x)
    x = Dropout(0.5)(x)

    # encoder 2
    for filters in [16, 32]:
        conv = UNet2DBlock(x, layers=2, filters=filters, dropout=0.2)
        x = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet2DBlock(x, layers=2, filters=64, dropout=0.5)
    # decoder 2
    for filters in [32, 16]:
        x = UNet2DBlock(x, layers=2, filters=filters, dropout=0.2)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet2DBlock(x, layers=2, filters=32, dropout=0.0)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    #focal_loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.2, gamma=2.0)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss = 'binary_crossentropy', metrics = [AUC(), dice_coef])

    model.summary()
    if weights is not None:
        model.load_weights(weights)
    return model


def experimental_network_3D(shape, weights=None):
    conv_encoder = []
    encoder_filters = np.array([8, 16, 32, 64, 64])
    decoder_filters = np.array([64, 64, 32, 16, 8])
    strides = (1, 1, 1)

    inputs = Input(shape)
    x = inputs
    
    # encoder
    for filters in encoder_filters:
        conv = UNet3DBlock(x, layers=2, filters=filters, strides=strides)
        x = MaxPooling3D(pool_size=(1, 2, 2))(conv)
        conv_encoder.append(conv)

    # bridge
    x = UNet3DBlock(x, layers=2, filters=64, dropout=0.5, strides=strides)

    # decoder
    for filters in decoder_filters:
        x = UNet3DBlock(x, layers=2, filters=filters, strides=strides)
        x = UpSampling3D(size=(1, 2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    #x = UNet3DBlock(x, layers=1, filters=8, strides=strides)
    # postcoder
    for operation, filters in zip([MaxPooling3D, MaxPooling3D, UpSampling3D, UpSampling3D], [8, 16, 16, 8]):
        x = UNet3DBlock(x, layers=1, filters=filters, dropout=0.25, strides=strides)
        x = operation((2, 2, 2))(x)

    outputs = Conv3D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=[AUC(), dice_coef])
    model.summary()

    return model



if __name__ == '__main__':
    pass
