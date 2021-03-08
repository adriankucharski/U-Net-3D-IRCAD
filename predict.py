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
from dataset import *


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
#                       PREDICTION 3D PATCH
# ************************************************************
def image_to_patch_generator(im, patch_size = 16, stride = 4):
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

                index = np.index_exp[left_w:right_w, left_h:right_h, left_d:right_d]
                yield im[index], index
    yield None, None

def predict_image_patch3D_generator(im, model, batch_size = 64, patch_size = 32, stride = 4):
    x, y, z = im.shape
    batch = np.zeros((batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float16)
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
        
        pred = model.predict(batch[0:batch_max], batch_size = 16, verbose=0)
        for i in range(batch_max):
            prob[indexes[i]] += pred[i]
            sums[indexes[i]] += 1
            
        if end:
            break
    return np.array(prob / sums, dtype=np.float32)


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


def predict_image_slice_and_save(im_path, im_save, model, static_size=None, mask=None):
    # load image and get array
    im = sitk.ReadImage(str(Path(im_path)))
    array = sitk.GetArrayFromImage(im)
    original_size = array.shape

    #multiply with mask
    

    # resize and preproces image
    if static_size is not None:
        array = resize_image(array, static_size)
        if mask is not None:
            mask = resize_image(mask, static_size)
            array[mask == 0] = np.mean(array[np.where(mask == 0)])
    
    #preprocess image
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

def predict_images_masked_slice(
    imgs_path, 
    model_path, 
    save_path='predicted', 
    im_name='patientIso.nii', 
    mask_name='dilatedVesselsMaskIso.nii', 
    static_size=(None, 224, 224)
):
    sp = Path(save_path)
    model = tf.keras.models.load_model(
        str(Path(model_path)), custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss, 'loss': my_loss()})

    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(imgs_path))), key=sorting):
        print(dir_path)
        ip = Path(dir_path) / im_name
        mp = Path(dir_path) / mask_name

        mask, _ = io_load_image(str(mp))
        suffix = Path(dir_path).parts[-1]

        save_name = sp / (suffix + '_patient.nii')
        predict_image_slice_and_save(
            ip, str(save_name), model, static_size, mask)

def predict_masked():
    MODEL = 'model\model_2D_24.01.2021_15-33-14.hdf5'
    predict_images_masked_slice('data/ircad_test/*', MODEL)
    exit()

if __name__ == '__main__':
    arr, im = io_load_image("data/ircad_test/3Dircadb1.15/patientIso.nii")
    arr = (arr - np.mean(arr)) / np.std(arr)

    model = tf.keras.models.load_model(
        str(Path('model/model_3D_03.03.2021_23-11-25.hdf5')), custom_objects={'dice_coef': dice_coef})

    a = predict_image_patch3D_generator(arr, model, patch_size = 16, stride=4)

    print(arr.shape, a.shape)
    io_save_image('data/test.nii', a, im)

    exit()
    MODEL = 'model\model_3D_18.02.2021_12-56-34.hdf5'
    IM_NAME = 'patientIso.nii'
    STATIC_SIZE = (None, 224, 224)
    predict_images_slab('data/ircad_snorkel/antiga098-002/test/*', MODEL, static_size=STATIC_SIZE, im_name=IM_NAME)

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
