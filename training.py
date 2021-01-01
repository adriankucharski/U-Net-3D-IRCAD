from dataset import *
from model import UNet3D
from History import History
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import resize
import tensorflow as tf
import SimpleITK as sitk
from datetime import datetime
import os
import pickle
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#####
#####

# ************************************************************
#                       TRAINING
# ************************************************************

def train_slab_unet3d(EPOCHES=150,
                      VALIDATION_SPLIT=0.2,
                      SLAB_PER_FILE=128,
                      SLAB_SHAPE=(16, 224, 224, 1),
                      dataset_path = 'data/im_data.pickle'):
    dataset = load_dataset(dataset_path)
    num_of_images = len(dataset)
    slabs_number = num_of_images * SLAB_PER_FILE

    X, Y = generate_slab_dataset(dataset, SLAB_PER_FILE, SLAB_SHAPE)
    model = UNet3D(SLAB_SHAPE)

    model_save = None
    history_save = None
    with open('model/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        model_save = Path('model') / ('model' + now + '.hdf5')
        history_save = Path('history') / ('history' + now + '.pickle')

        log.write('######################################\n')
        log.write('Model path       : ' + str(model_save) + ':\n')
        log.write('History          : ' + str(history_save) + '\n')
        log.write('Epoches          : ' + str(EPOCHES) + '\n')
        log.write('Validation split : ' + str(VALIDATION_SPLIT) + '\n')
        log.write('Slabs per image  : ' + str(SLAB_PER_FILE) + '\n')
        log.write('Slab shape       : ' + str(SLAB_SHAPE) + '\n')
        log.write('Number of slabs  : ' + str(slabs_number) + '\n')
        log.write('Dataset          : ' + dataset_path + '\n')

    checkpointer = ModelCheckpoint(
            str(model_save), 'val_loss', 2, True, mode='auto')
    history = model.fit(X, Y, verbose=1, epochs=EPOCHES, batch_size=1,
                            validation_split=VALIDATION_SPLIT, callbacks=[checkpointer])
    hist = History(history)
    hist.save_history(history_save)

    return model_save
