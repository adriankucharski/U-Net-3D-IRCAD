
import os
import pickle
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime

import SimpleITK as sitk
import tensorflow as tf
from skimage.transform import resize

#####
from dataset import resize_image, preproces_gt, preproces_im
#####

# ************************************************************
#                       PREDICTION
# ************************************************************


def predict_image_slab(im, model, slice_per_slab: int = 16, stride: int = 1):
    predict = np.zeros(im.shape, np.float16)
    count = np.zeros(im.shape, np.float16)

    for ID in range(0, im.shape[1] - slice_per_slab + 1, stride):
        index = np.index_exp[:, ID:ID+slice_per_slab, :, :]
        slab = im[index]
        output = model.predict(slab, verbose=0)
        predict[index] += output[:, :, :, :, 0:1]
        count[index] += 1

    count[np.where(count == 0)] = 1
    predict = predict / count
    return predict


def predict_image_slab_and_save(im_path, im_save, model, slice_per_slab: int = 16, stride: int = 1, static_size=None):
    # load image and get array
    im = sitk.ReadImage(str(Path(im_path)))
    array = sitk.GetArrayFromImage(im)
    original_size = array.shape

    # resize and preproces image
    if static_size is not None:
        array = resize_image(array, static_size)
    array = preproces_im(array)

    new_im = predict_image_slab(array, model, slice_per_slab, stride)
    new_im = np.array(np.reshape(
        new_im, (new_im.shape[1], new_im.shape[2], new_im.shape[3])), dtype=np.float)
    new_im = np.array(resize(new_im, original_size) * 255, dtype=np.uint8)

    #newIm = (newIm > threshold_li(newIm)) * 255

    new_im = sitk.GetImageFromArray(new_im)
    new_im.CopyInformation(im)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(Path(im_save)))
    writer.Execute(new_im)


def predict_images_slab(imgs_path, model_path, save_path='predicted', im_name='patientIso.nii', slice_per_slab: int = 16, stride: int = 1, static_size=(None, 224, 224)):
    sp = Path(save_path)
    model = tf.keras.models.load_model(
        str(Path(model_path)), custom_objects={'None': None})

    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(imgs_path))), key=sorting):
        print(dir_path)
        ip = Path(dir_path) / im_name
        suffix = Path(dir_path).parts[-1]

        save_name = sp / (suffix + '_patient.nii')
        predict_image_slab_and_save(
            ip, str(save_name), model, slice_per_slab, stride, static_size)
