import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset import resize_image, preproces_gt, preproces_im
from skimage.transform import resize
import tensorflow as tf
import SimpleITK as sitk
from datetime import datetime
import pickle
import random
import re
from glob import glob
from pathlib import Path
from model import *

import numpy as np
from skimage import io



#####
#####

# ************************************************************
#                       PREDICTION 3D
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
        str(Path(model_path)), custom_objects={'dice_coef': dice_coef})
    model.summary()

    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(imgs_path))), key=sorting):
        print(dir_path)
        ip = Path(dir_path) / im_name
        suffix = Path(dir_path).parts[-1]

        save_name = sp / (suffix + '_patient.nii')
        predict_image_slab_and_save(
            ip, str(save_name), model, slice_per_slab, stride, static_size)


# ************************************************************
#                       PREDICTION 2D
# ************************************************************
def predict_image_slice(im, model):
    shape = im.shape
    pred = np.ones(shape, dtype=np.float16)
    for idx in range(0, im.shape[1]):
        part = im[:, idx, :, :]
        pred[0, idx, :, :, :] = model.predict(part, batch_size=1)
    return pred


def predict_image_slice_and_save(im_path, im_save, model, static_size=None):
    # load image and get array
    im = sitk.ReadImage(str(Path(im_path)))
    array = sitk.GetArrayFromImage(im)
    original_size = array.shape

    # resize and preproces image
    if static_size is not None:
        array = resize_image(array, static_size)
    array = preproces_im(array)

    new_im = predict_image_slice(array, model)
    new_im = np.array(np.reshape(
        new_im, (new_im.shape[1], new_im.shape[2], new_im.shape[3])), dtype=np.float)
    new_im = np.array(resize(new_im, original_size) * 255, dtype=np.uint8)

    #newIm = (newIm > threshold_li(newIm)) * 255

    new_im = sitk.GetImageFromArray(new_im)
    new_im.CopyInformation(im)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(Path(im_save)))
    writer.Execute(new_im)


def predict_images_slice(imgs_path, model_path, save_path='predicted', im_name='patientIso.nii', static_size=(None, 224, 224)):
    sp = Path(save_path)
    model = tf.keras.models.load_model(
        str(Path(model_path)), custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss, 'loss': my_loss()})

    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(imgs_path))), key=sorting):
        print(dir_path)
        ip = Path(dir_path) / im_name
        suffix = Path(dir_path).parts[-1]

        save_name = sp / (suffix + '_patient.nii')
        predict_image_slice_and_save(
            ip, str(save_name), model, static_size)


if __name__ == '__main__':
    MODEL = 'model\model_2D_23.01.2021_15-58-30.hdf5'
    STATIC_SIZE = (None, 224, 224)
    predict_images_slice('data/ircad_test/*', MODEL, static_size=STATIC_SIZE)

"""
predicted\3Dircadb1.15_patient.nii 0.9403078036832394
predicted\3Dircadb1.1_patient.nii 0.9440268308848263
predicted\3Dircadb1.20_patient.nii 0.9505963172274714
predicted\3Dircadb1.5_patient.nii 0.9664644101758728

predicted\3Dircadb1.15_patient.nii 0.9328859469940245
predicted\3Dircadb1.1_patient.nii 0.9566474943916878
predicted\3Dircadb1.20_patient.nii 0.9435723972586476
predicted\3Dircadb1.5_patient.nii 0.9652332296180478
"""