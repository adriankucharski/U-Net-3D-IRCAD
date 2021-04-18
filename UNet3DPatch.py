import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Accuracy, MeanIoU, Precision, Recall
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv3D, Dropout, Input, LeakyReLU,
                                     MaxPooling2D, MaxPooling3D, Softmax,
                                     UpSampling2D, UpSampling3D, concatenate)
from tensorflow.keras.activations import sigmoid, softmax, tanh
from tensorflow.keras import backend as K
from skimage.transform import resize
from skimage import io
import tensorflow_addons as tfa
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
import re
import random
import pickle
from History import History
from threading import Thread 


###################################################
#                   Metrics
###################################################
def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = 2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return intersection / sums

def dice_loss(y_true, y_pred):
    smooth=1e-5
    intersection = 2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return 1.0 - (intersection / sums)


###################################################
#                   Model
###################################################
def UNet3DBlock(inputs, layers, filters, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', dropout=0.25, strides=1):
    conv = inputs
    for _ in range(layers):
        conv = Conv3D(filters, kernel_size, activation=activation,
                      padding=padding, kernel_initializer=kernel_initializer, strides=strides)(conv)
    conv = Dropout(dropout)(conv)
    return conv


def UNet3DPatch(shape, weights=None):
    conv_encoder = []
    encoder_filters = np.array([8, 16, 32, 32])
    mid_filters = 32
    decoder_filters = np.array([64, 32, 16, 8])
    bottom_filters = 4

    inputs = Input(shape)
    # encoder
    x = inputs
    for filters in encoder_filters:
        conv = UNet3DBlock(x, layers=2, filters=filters)
        x = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet3DBlock(x, layers=1, filters=mid_filters)

    # decoder
    for filters in decoder_filters:
        x = UNet3DBlock(x, layers=2, filters=filters)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet3DBlock(x, layers=2, filters=bottom_filters)
    outputs = Conv3D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=[AUC(), dice_coef])
    model.summary()

    return model

###################################################
#                   Preproces
###################################################
def preproces_im(im) -> np.ndarray:
    im = im / np.max(im)
    #im = (im - np.mean(im)) / np.std(im)
    return np.array(np.reshape(im, (-1, *im.shape, 1)), dtype='float16')

def preproces_gt(gt):
    gt[gt > 0] = 1.0
    return np.reshape(gt, (-1, *gt.shape, 1))


###################################################
#                   IO
###################################################
def io_load_image(path: str) -> list:
    image = sitk.ReadImage(str(Path(path)))
    array = sitk.GetArrayFromImage(image)
    return [array, image]


def io_save_image(path: str, array, image_header_data):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(image_header_data)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(Path(path)))
    writer.Execute(image)


def load_dataset(
    from_path: str = 'final_dataset/dataset_1/train/*',
    im_name: str = 'patientIso.nii',
    gt_name: str = 'liverMaskIso.nii'
) -> tuple:
    im_data = []
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for im_dir in sorted(glob(str(Path(from_path))), key=sorting):
        im_path = Path(im_dir) / im_name
        gt_path = Path(im_dir) / gt_name

        x, _ = io_load_image(str(im_path))
        y, _ = io_load_image(str(gt_path))

        x = preproces_im(x).astype('float16')
        y = preproces_gt(y).astype('float16')
        im_data.append((x, y))

    return im_data

###################################################
#                 Patch generation
###################################################
def create_patch_training_data(dataset, patch_size=16, patch_per_file=10000):
    num_of_files = len(dataset)

    patch_shape = (patch_size, patch_size, patch_size, 1)
    X = np.empty((patch_per_file * num_of_files,
                  *patch_shape), dtype=np.float16)
    Y = np.empty((patch_per_file * num_of_files,
                  *patch_shape), dtype=np.float16)

    patch_index = 0
    for pair in dataset:
        x, y = pair
        mid = patch_size//2
        _, _x, _y, _z, _ = x.shape
        corr_x = np.random.randint(mid, _x - mid - 1, patch_per_file)
        corr_y = np.random.randint(mid, _y - mid - 1, patch_per_file)
        corr_z = np.random.randint(mid, _z - mid - 1, patch_per_file)

        for i in np.arange(0, patch_per_file):
            # example index with patch_size = 32: [0, 0:32, 45:77, 100:132]
            index = np.index_exp[0, corr_x[i] - mid: corr_x[i] + mid,
                                 corr_y[i] - mid: corr_y[i] + mid,  corr_z[i] - mid: corr_z[i] + mid]
            X[patch_index, :, :, :] = x[index]
            Y[patch_index, :, :, :] = y[index]
            patch_index += 1
    return X, Y

def patch_generator_data(dataset, patch_size = 16, batch_size = 32):
    num_of_files = len(dataset)
    patch_shape = (patch_size, patch_size, patch_size, 1)

    while True:
        idx = random.randint(0, len(dataset) - 1)

        mid = patch_size//2
        x, y = dataset[idx]
        _, _x, _y, _z, _ = x.shape
        corr_x = np.random.randint(mid, _x - mid - 1, batch_size)
        corr_y = np.random.randint(mid, _y - mid - 1, batch_size)
        corr_z = np.random.randint(mid, _z - mid - 1, batch_size)

        for i in np.arange(0, batch_size):
            index = np.index_exp[0, corr_x[i] - mid: corr_x[i] + mid,
                                 corr_y[i] - mid: corr_y[i] + mid,  corr_z[i] - mid: corr_z[i] + mid]

            yield np.expand_dims(x[index], axis=0), np.expand_dims(y[index], axis=0)


# ************************************************************
#                         predict
# ************************************************************
def image_to_patch_generator(im, patch_size=16, stride=4):
    W, H, D = im.shape
    im = np.reshape(im, (*im.shape, 1))

    W = (W - patch_size)//stride + 1
    H = (H - patch_size)//stride + 1
    D = (D - patch_size)//stride + 1
    for w in range(W):
        for h in range(H):
            for d in range(D):
                left_w, right_w = w*stride, w*stride+patch_size
                left_h, right_h = h*stride, h*stride+patch_size
                left_d, right_d = d*stride, d*stride+patch_size

                index = np.index_exp[left_w:right_w,
                                     left_h:right_h, left_d:right_d]
                yield im[index], index
    yield None, None

def predict_image(im, model, batch_size=64, patch_size=16, stride=4):
    x, y, z = im.shape
    batch = np.zeros((batch_size, patch_size, patch_size,
                      patch_size, 1), dtype=np.float16)
    indexes = list(range(batch_size))

    prob = np.zeros((x, y, z, 1), np.float16)
    sums = np.ones((x, y, z, 1), np.float16)

    gen = image_to_patch_generator(im, patch_size, stride)
    while True:
        end = False
        batch_max = 0
        for i in range(batch_size):
            patch, index = next(gen)
            if patch is None:
                end = True
                break
            batch[i], indexes[i] = patch, index
            batch_max += 1
        
        if batch_max == 0:
            break

        pred = model.predict(batch[0:batch_max], batch_size=16, verbose=0)
        for i in range(batch_max):
            prob[indexes[i]] += pred[i]
            sums[indexes[i]] += 1

        if end:
            break
    return np.array(prob / sums, dtype=np.float32)

def gen_crossvalidation_array(num_of_el: int):
    a = list(range(num_of_el))
    cross = []
    for i in range(len(a)):
        test = i
        train = list(a)
        train.remove(test)
        cross.append([train, test])
    return cross

# ************************************************************
#                         main
# ************************************************************
if __name__ == '__main__':
    epochs = 60
    batch_size = 128
    steps_per_epoch = 1250
    model_path_save = 'model/cross/model_3d_patch.hdf5'
    history_path_save = 'history/cross/history_'

    patch_size = 32
    dataset_path = 'data/ircad_full/*'


    ############################################
    # Load dataset
    dataset = load_dataset(dataset_path)
    cross = gen_crossvalidation_array(len(dataset))

    for train, test in cross:
        if test < 10:
            continue
        ############################################
        # Get the train fold
        print(train, test)
        model_path_save_2 = './' + model_path_save + str(test)
        cross_train = [dataset[i] for i in train]

        ############################################
        # Training
        checkpointer = ModelCheckpoint(
            model_path_save_2, 'loss', 2, True, mode='auto')
        model = UNet3DPatch((patch_size, patch_size, patch_size, 1))

        gen = patch_generator_data(cross_train, patch_size, batch_size)
        hist = model.fit(gen, batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpointer])


        ############################################
        # Save history
        h_path = history_path_save + '.hist_' + str(test) 
        History(hist).save_history(h_path)

        
        ############################################
        # Predict
        trained_model = tf.keras.models.load_model(
            str(Path(model_path_save_2)), custom_objects={'dice_coef': dice_coef, 'dice_loss:' : dice_loss})

        def sorting(s): return int(re.findall(r'\d+', s)[-1])
        for im_dir in sorted(glob(str(Path(dataset_path))), key=sorting):
            if ('.' + str(test + 1)) not in im_dir:
                continue

            im_path = Path(im_dir) / 'patientIso.nii'
            im, im_data = io_load_image(str(im_path))

            im = preproces_im(im)
            _, xx, yy, zz, _ = im.shape
            im = np.reshape(im, (xx, yy, zz))

            print('Predict: ', im_path)
            im = predict_image(im, trained_model, batch_size, patch_size, 4)

            p = 'predicted/cross/'
            io_save_image(p + im_path.parts[-2] + '.nii', im, im_data)
            break
        
"""
@ Could you remind me how did you train your networks? You used low resolution images, resized all of them to 256x256x? 
"""
