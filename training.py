import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from dataset import *
from model import UNet3D, UNet3DBlock, UNet2D, UNet2DBlock, UNet3DPatch
from History import History
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import resize
import tensorflow as tf
import SimpleITK as sitk
from datetime import datetime

import pickle
import random
import re
from glob import glob
from pathlib import Path
import inspect


import numpy as np
from skimage import io


#####
#####

# ************************************************************
#                       TRAINING 3D
# ************************************************************

def train_slab_unet3d(EPOCHES=150,
                      VALIDATION_SPLIT=0.2,
                      SLAB_PER_FILE=128,
                      SLAB_SHAPE=(16, 224, 224, 1),
                      dataset_path='data/im_data.pickle',
                      BATCH_SIZE=1,
                      NETWORK=UNet3D):
    dataset = load_dataset(dataset_path)
    num_of_images = len(dataset)
    slabs_number = num_of_images * SLAB_PER_FILE

    X, Y = generate_slab_dataset(dataset, SLAB_PER_FILE, SLAB_SHAPE)
    model = NETWORK(SLAB_SHAPE)

    model_save = None
    history_save = None
    with open('model/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        model_save = Path('model') / ('model_3D_' + now + '.hdf5')
        history_save = Path('history') / ('history' + now + '.pickle')

        log.write('\n################### 3D #################\n')
        log.write('Model path       : ' + str(model_save) + '\n')
        log.write('History          : ' + str(history_save) + '\n')
        log.write('Epoches          : ' + str(EPOCHES) + '\n')
        log.write('Validation split : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Batch size       : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Slabs per image  : ' + str(SLAB_PER_FILE) + '\n')
        log.write('Slab shape       : ' + str(SLAB_SHAPE) + '\n')
        log.write('Number of slabs  : ' + str(slabs_number) + '\n')
        log.write('Dataset          : ' + dataset_path + '\n')
        log.write('Code             : [\n')
        log.write(inspect.getsource(UNet3DBlock) + '\n')
        log.write(inspect.getsource(NETWORK) + '\n]\n')

    checkpointer = ModelCheckpoint(
        str(model_save), 'val_loss', 2, True, mode='auto')
    history = model.fit(X, Y, verbose=1, epochs=EPOCHES, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, callbacks=[checkpointer])
    hist = History(history)
    hist.save_history(history_save)

    return model_save

def train_patch_unet3d(EPOCHES=150,
                      VALIDATION_SPLIT=0.2,
                      PATCH_PER_FILE = 100,
                      PATCH_SIZE = 32,
                      dataset_path='data/im_data.pickle',
                      BATCH_SIZE=1,
                      NETWORK=UNet3DPatch):
    dataset = load_dataset(dataset_path)
    X, Y = get_patch_dataset(dataset, PATCH_SIZE, PATCH_PER_FILE)
    model = NETWORK((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))

    model_save = None
    history_save = None
    with open('model/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        model_save = Path('model') / ('model_3D_' + now + '.hdf5')
        history_save = Path('history') / ('history' + now + '.pickle')

        log.write('\n################### 3D #################\n')
        log.write('Model path       : ' + str(model_save) + '\n')
        log.write('History          : ' + str(history_save) + '\n')
        log.write('Epoches          : ' + str(EPOCHES) + '\n')
        log.write('Validation split : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Batch size       : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Patch per image  : ' + str(PATCH_PER_FILE) + '\n')
        log.write('Patch size       : ' + str(PATCH_SIZE) + '\n')
        log.write('Dataset          : ' + dataset_path + '\n')
        log.write('Code             : [\n')
        log.write(inspect.getsource(UNet3DBlock) + '\n')
        log.write(inspect.getsource(NETWORK) + '\n]\n')

    checkpointer = ModelCheckpoint(
        str(model_save), 'val_loss', 2, True, mode='auto')
    history = model.fit(X, Y, verbose=1, epochs=EPOCHES, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, callbacks=[checkpointer])
    hist = History(history)
    hist.save_history(history_save)

    return model_save

# ************************************************************
#                       TRAINING 2D
# ************************************************************


def train_slice_unet2d(EPOCHES=150,
                       VALIDATION_SPLIT=0.2,
                       SLICE_PER_FILE=128,
                       SLICE_SHAPE=(224, 224, 1),
                       dataset_path='data/im_data.pickle',
                       BATCH_SIZE=32,
                       network=UNet2D):
    dataset = load_dataset(dataset_path)
    num_of_images = len(dataset)
    slices_number = num_of_images * SLICE_PER_FILE

    X, Y = None, None
    if SLICE_PER_FILE == -1:
        X, Y = get_slice_dataset(dataset, SLICE_SHAPE)
    else:    
        X, Y = generate_slice_dataset(dataset, SLICE_PER_FILE, SLICE_SHAPE)
    
    assert(X is not None and Y is not None)

    print("Dataset max values: ",
          "[X_min=", X.min(), 'X_max=', X.max(), ']', "Ymax=", Y.max())
    model = network(SLICE_SHAPE)

    model_save = None
    history_save = None
    with open('model/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        model_save = Path('model') / ('model_2D_' + now + '.hdf5')
        history_save = Path('history') / ('history' + now + '.pickle')

        log.write('\n################## 2D ##################\n')
        log.write('Model path       : ' + str(model_save) + '\n')
        log.write('History          : ' + str(history_save) + '\n')
        log.write('Epoches          : ' + str(EPOCHES) + '\n')
        log.write('Validation split : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Batch size       : ' + str(BATCH_SIZE) + '\n')
        log.write('Slice per image  : ' + str(SLICE_PER_FILE) + '\n')
        log.write('Slice shape      : ' + str(SLICE_SHAPE) + '\n')
        log.write('Number of slices : ' + str(slices_number) + '\n')
        log.write('Dataset          : ' + dataset_path + '\n')
        log.write('Code             : [\n')
        log.write(inspect.getsource(UNet2DBlock) + '\n')
        log.write(inspect.getsource(network) + '\n]\n')

    checkpointer = ModelCheckpoint(
        str(model_save), 'val_loss', 2, True, mode='auto')
    history = model.fit(X, Y, verbose=1, epochs=EPOCHES, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, callbacks=[checkpointer])
    hist = History(history)
    hist.save_history(history_save)

    return model_save


def train_slice_generator_unet2d(EPOCHES=150,
                                 SLICE_PER_FILE=128,
                                 VALIDATION_SPLIT=0.2,
                                 SLICE_SHAPE=(224, 224, 1),
                                 dataset_path='data/im_data.pickle',
                                 BATCH_SIZE=32,
                                 network=UNet2D,
                                 step_per_epoch=60):

    GEN, VALID, step_per_epoch = get_slice_generator(
        dataset_path, SLICE_PER_FILE, SLICE_SHAPE, VALIDATION_SPLIT, BATCH_SIZE)

    X_valid, Y_valid = VALID
    print("Dataset max values: ", "[X_min=", X_valid.min(
    ), 'X_max=', X_valid.max(), ']', "Ymax=", Y_valid.max())
    model = network(SLICE_SHAPE)

    model_save = None
    history_save = None
    with open('model/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        model_save = Path('model') / ('model_2D_' + now + '.hdf5')
        history_save = Path('history') / ('history' + now + '.pickle')

        log.write('\n################## 2D GENERATOR ##################\n')
        log.write('Model path       : ' + str(model_save) + ':\n')
        log.write('History          : ' + str(history_save) + '\n')
        log.write('Epoches          : ' + str(EPOCHES) + '\n')
        log.write('Validation split : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Batch size       : ' + str(BATCH_SIZE) + '\n')
        log.write('Slice per image  : ' + str(SLICE_PER_FILE) + '\n')
        log.write('Slice shape      : ' + str(SLICE_SHAPE) + '\n')
        log.write(' Step per epoch  : ' + str(step_per_epoch) + '\n')
        log.write('Dataset          : ' + dataset_path + '\n')
        log.write('Code             : [\n')
        log.write(inspect.getsource(UNet2DBlock) + '\n')
        log.write(inspect.getsource(network) + '\n]\n')

    checkpointer = ModelCheckpoint(
        str(model_save), 'val_loss', 2, True, mode='auto')
    history = model.fit(GEN, verbose=1, epochs=EPOCHES, steps_per_epoch=step_per_epoch/BATCH_SIZE,
                        validation_data=(X_valid, Y_valid), validation_batch_size=BATCH_SIZE, callbacks=[checkpointer])
    hist = History(history)
    hist.save_history(history_save)

    return model_save
