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
from skimage.transform import resize

# ************************************************************
#                       DATASET
# ************************************************************

def generate_slab_dataset(dataset, slab_per_file: int = 128, slab_shape=(16, 224, 224, 1)):
    num_of_files = len(dataset)
    slabs = num_of_files * slab_per_file
    X = np.empty((slabs, *slab_shape), dtype=np.float16)
    Y = np.empty((slabs, *slab_shape), dtype=np.float16)

    slab_index = 0
    for image_id in range(num_of_files):
        i = 0
        IDs = []
        x, y = dataset[image_id]
        while i < slab_per_file:
            slab_id = int(np.random.randint(0, x.shape[1]))
            if slab_id not in IDs:
                index = np.index_exp[0, slab_id, :, :]
                X[slab_index, :, :, :, :] = x[index]
                Y[slab_index, :, :, :, :] = y[index]
                IDs.append(slab_id)
                slab_index += 1
                i += 1
    return X, Y

#[128, 224, 224, 1]
#[1, Z, 224, 224, 1]

def generate_slice_dataset(dataset, slice_per_file:int = 128, slice_shape=(224, 224, 1)):
    num_of_files = len(dataset)
    slices = num_of_files * slice_per_file
    X = np.empty((slices, *slice_shape), dtype=np.float16)
    Y = np.empty((slices, *slice_shape), dtype=np.float16)

    slice_index = 0
    for image_id in range(num_of_files):
        i = 0
        IDs = []
        x, y = dataset[image_id]
        while i < slice_per_file:
            slice_id = int(np.random.randint(0, x.shape[1]))
            if slice_id not in IDs:
                index = np.index_exp[0, slice_id, :, :]
                X[slice_index, :, :, :] = x[index]
                Y[slice_index, :, :, :] = y[index]
                IDs.append(slice_id)
                slice_index += 1
                i += 1
    return X, Y



def resize_image(data, static_size: tuple):
    new_size = list(data.shape)
    for i in range(len(static_size)):
        new_size[i] = new_size[i] if static_size[i] == None else static_size[i]

    data_resized = resize(data, new_size, anti_aliasing=False)
    return np.array(data_resized, dtype=np.float16)


def preproces_im(im):
    im = (im - np.mean(im)) / np.std(im)
    im = np.reshape(im, (*im.shape, 1))
    new_im = np.zeros((1, *im.shape), dtype=np.float16)
    new_im[:, :, :, :] = im
    return new_im


def preproces_gt(gt, label=(0, 1)):
    depth, height, width = gt.shape
    gt = np.reshape(gt, (depth, height, width)) 
    new_gt = np.zeros((depth, height, width, len(label)), dtype='float16')
    print("PG", np.max(gt))
    for i in range(0, len(label)):
        new_gt[0:depth, 0:height, 0:width, i] = (gt == label[i])

    reshaped = np.zeros((1, *new_gt.shape), dtype=np.float16)
    reshaped[:, :, :, :] = new_gt
    return reshaped


def load_pair_images(im_path, gt_path) -> list:
    X = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(im_path))))
    Y = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(gt_path))))
    return [X, Y]


def prepare_dataset(from_path='data/ircad_iso_111/*', im_name='patientIso.nii', gt_name='liverMaskIso.nii', save_path='data/im_data.pickle', static_size=None):
    im_data = []
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(from_path))), key=sorting):
        print(dir_path)

        ip = Path(dir_path) / im_name
        gp = Path(dir_path) / gt_name
        X, Y = load_pair_images(ip, gp)
        

        if static_size is not None:
            X = resize_image(X, static_size)
            Y = resize_image(Y, static_size)

        X = preproces_im(X)
        Y = preproces_gt(Y, list([Y.max()]))
        print("Max: ", np.max(Y))
        im_data.append((X, Y))

    with open('data/log.txt', 'a+') as log:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        num_of_images = len(im_data)

        im_size = 'original'
        if static_size is not None:
            im_size = [
                str(x) if x is not None else 'original' for x in static_size]

        log.write(now + '\n')
        log.write('\t Save path: ' + save_path + '\n')
        log.write('\t Image name: ' + im_name + '\n')
        log.write('\t Gt name: ' + gt_name + '\n')
        log.write('\t Number of training images: ' + str(num_of_images) + '\n')
        log.write('\t Image size: ' + str(im_size) + '\n')
        log.write('\n')

    save_dataset(im_data, save_path)


def save_dataset(im_data, save_path: str = 'data/im_data.pickle'):
    with open(str(Path(save_path)), 'wb') as file:
        pickle.dump(im_data, file)


def load_dataset(path: str = 'data/im_data.pickle') -> tuple:
    im_data = None
    with open(str(Path(path)), 'rb') as file:
        im_data = pickle.load(file)
    return im_data

if __name__ == '__main__':
    prepare_dataset(from_path = 'data/ircad_iso_111/*', gt_name = 'liverMaskIso.nii', save_path='data/im_liver_data.pickle', static_size=(None, 224, 224))
    dataset = load_dataset('data/im_liver_data.pickle')
    X, Y = generate_slice_dataset(dataset)
    print(np.max(Y), np.max(X))
    print(X.shape, Y.shape)
