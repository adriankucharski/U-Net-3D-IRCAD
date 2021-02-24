import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skimage import io
import numpy as np
from pathlib import Path
from glob import glob
import re
import random
import pickle
from datetime import datetime
import SimpleITK as sitk
from skimage.transform import resize
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



# ************************************************************
#                       IMAGE IO
# ************************************************************

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


def io_load_pair_images(im_path, gt_path) -> list:
    X = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(im_path))))
    Y = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(gt_path))))
    return [X, Y]


def save_dataset(im_data, save_path: str = 'data/im_data.pickle'):
    with open(str(Path(save_path)), 'wb') as file:
        pickle.dump(im_data, file)


def load_dataset(path: str = 'data/im_data.pickle') -> tuple:
    im_data = None
    with open(str(Path(path)), 'rb') as file:
        im_data = pickle.load(file)
    return im_data


# ************************************************************
#                       PREPROCESS
# ************************************************************
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def resize_image(data, static_size: tuple):
    new_size = list(data.shape)
    for i in range(len(static_size)):
        new_size[i] = new_size[i] if static_size[i] == None else static_size[i]

    data_resized = resize(data, new_size, anti_aliasing=False)
    return np.array(data_resized, dtype=np.float16)

def preproces_im(im):
    im = (im - np.mean(im)) / np.std(im)
    #im = sigmoid(im)
    #im = np.tanh(im)
    return np.reshape(im, (-1, *im.shape, 1))


def preproces_gt(gt, label=(0, 1)):
    depth, height, width = gt.shape
    gt = np.reshape(gt, (depth, height, width))
    new_gt = np.zeros((depth, height, width, len(label)), dtype='float16')
    for i in range(0, len(label)):
        new_gt[0:depth, 0:height, 0:width, i] = (gt == label[i])

    return np.reshape(new_gt, (-1, *new_gt.shape))


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


def generate_slice_dataset(dataset, slice_per_file: int = 128, slice_shape=(224, 224, 1)):
    num_of_files = len(dataset)
    slices = num_of_files * slice_per_file
    X = np.empty((slices, *slice_shape), dtype=np.float16)
    Y = np.empty((slices, *slice_shape), dtype=np.float16)

    slice_index = 0
    for image_id in range(num_of_files):
        i = 0
        IDs = []
        x, y = dataset[image_id]
        while i < slice_per_file and i < len(x):
            slice_id = int(np.random.randint(0, x.shape[1]))
            if slice_id not in IDs:
                index = np.index_exp[0, slice_id, :, :]
                X[slice_index, :, :, :] = x[index]
                Y[slice_index, :, :, :] = y[index]
                IDs.append(slice_id)
                slice_index += 1
                i += 1
    return X, Y


def get_slice_dataset(dataset, slice_shape=(224, 224, 1)):
    assert (len(slice_shape) == 3)

    num_of_files = len(dataset)
    slices = np.sum([pack[0].shape[1] for pack in dataset])
    print(slices)
    X = np.empty((slices, *slice_shape), dtype=np.float16)
    Y = np.empty((slices, *slice_shape), dtype=np.float16)

    slice_index = 0
    for pair in dataset:
        x, y = pair
        print(x.shape, y.shape)
        for i in range(x.shape[1]):
            index = np.index_exp[0, i, :, :]
            X[slice_index, :, :, :] = x[index]
            Y[slice_index, :, :, :] = y[index]
            slice_index += 1
    return X, Y

def get_patch_dataset(dataset, patch_size = 32, patch_per_file = 200):
    num_of_files = len(dataset)

    patch_shape = (patch_size, patch_size, patch_size, 1)
    X = np.empty((patch_per_file * num_of_files, *patch_shape), dtype = np.float16)
    Y = np.empty((patch_per_file * num_of_files, *patch_shape), dtype = np.float16)

    patch_index = 0
    for pair in dataset:
        x, y = pair
        mid = patch_size//2
        _, _x, _y, _z, _ = x.shape
        corr_x = np.random.permutation(np.arange(mid, _x - mid))[0:patch_per_file]
        corr_y = np.random.permutation(np.arange(mid, _y - mid))[0:patch_per_file]
        corr_z = np.random.permutation(np.arange(mid, _z - mid))[0:patch_per_file]

        for i in np.arange(0, patch_per_file):
            #example index with patch_size = 32: [0, 0:32, 45:77, 100:132]
            index = np.index_exp[0, corr_x[i] - mid : corr_x[i] + mid, corr_y[i] - mid : corr_y[i] + mid,  corr_z[i] - mid : corr_z[i] + mid]
            X[patch_index, :, :, :] = x[index]
            Y[patch_index, :, :, :] = y[index]
            patch_index += 1
    return X, Y



def prepare_dataset(from_path='data/ircad_iso_111/*', im_name='patientIso.nii', gt_name='liverMaskIso.nii', save_path='data/im_data.pickle', static_size=None):
    im_data = []
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(from_path))), key=sorting):
        ip = Path(dir_path) / im_name
        gp = Path(dir_path) / gt_name
        X, Y = io_load_pair_images(ip, gp)

        if static_size is not None:
            X = resize_image(X, static_size)
            Y = resize_image(Y, static_size)

        X = preproces_im(X)
        Y = preproces_gt(Y, list([Y.max()]))
        print(dir_path, "Max Y: ", np.max(Y),
              "[Min X: ", np.min(X), " Max X: ", np.max(X), "]")
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


def get_slice_generator(path: str = 'data/im_data.pickle', slice_per_file: int = 128, slice_shape=(224, 224, 1), val_split: float = 0.2, batch_size: int = 32) -> tuple:
    # Load dataset from .pickle file
    dataset = load_dataset(path)

    # Extract slice from 3D img
    X, Y = generate_slice_dataset(dataset, slice_per_file, slice_shape)

    # Shuffle arrays
    p = np.random.permutation(len(X))
    X, Y = X[p], Y[p]

    # Extract validation data
    size_valid = int(len(X) * val_split)
    X_valid, Y_valid = X[0:size_valid], Y[0:size_valid]
    X_train, Y_train = X[size_valid:len(X)], Y[size_valid:len(Y)]

    # Create generator
    datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        dtype=np.float16
    )
    datagen.fit(X)

    data_generator = datagen.flow(X_train, Y_train, batch_size)
    data_validation = (X_valid, Y_valid)
    x_train_size = len(X_train)
    return data_generator, data_validation, x_train_size

########################################################################
#                         DATASET MANIPULATION                         #
########################################################################


def extract_mask_area_from_images(from_path='data/ircad_train/*', im_name='patientIso.nii', gt_name='dilatedVesselsMaskIso.nii', save_path='maskedDilatedVesselsIso.nii'):
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(from_path))), key=sorting):
        ip = Path(dir_path) / im_name
        gp = Path(dir_path) / gt_name

        im, data = io_load_image(ip)
        gt, _ = io_load_image(gp)

        im[gt == 0] = np.min(im)
        print(str(Path(dir_path) / save_path))
        io_save_image(str(Path(dir_path) / save_path), im, data)


def get_extracted_dataset(
    from_path='data/ircad_train/*', 
    im_name='patientIso.nii', 
    gt_name='vesselsIso.nii',
    mask_name='dilatedVesselsMaskIso.nii', 
    save_path='maskedDilatedVesselsIso.nii', 
    static_size=(None, 224, 224)
):
    im_data = []
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    for dir_path in sorted(glob(str(Path(from_path))), key=sorting):
        ip = Path(dir_path) / im_name
        gp = Path(dir_path) / gt_name
        mp = Path(dir_path) / mask_name
        X, Y = io_load_pair_images(ip, gp)
        M, _ = io_load_image(mp)

        if static_size is not None:
            X = resize_image(X, static_size)
            Y = resize_image(Y, static_size)
            M = resize_image(M, static_size)

        X = preproces_im(X)
        Y = preproces_gt(Y, list([Y.max()]))
        M = preproces_gt(M, list([Y.max()]))
        X[M == 0] = np.min(X[np.where(M == 0)])
        print(dir_path, "Max Y: ", np.max(Y),
              "[Min X: ", np.min(X), " Max X: ", np.max(X), "]", np.sum(np.where(M == 0)))
        im_data.append((X, Y))
    save_dataset(im_data, 'data/im_vessels_masked_16train_4test.pickle')

import matplotlib.pyplot as plt
def show_dataset(index: int = 0):
    name = 'data/im_vessels_masked_16train_4test.pickle'
    dataset = load_dataset(name)
    x, y = dataset[0]

    im = np.array((x[0, index, :, :] + 1) / 2.0 * 255, dtype=np.int)
    print(im.shape)
    plt.imshow(im, cmap='gray')
    plt.show()
    exit()

if __name__ == '__main__':
    shape = (224, 224)
    name = 'data/im_antiga098_002_masked_16train_4test.pickle'
    # from_path = 'data/ircad_snorkel/antiga098-002/train/*'
    # gt_name = 'antiga.nii'
    # im_name = 'patientIso.nii'
    # prepare_dataset(from_path=from_path, im_name=im_name, gt_name=gt_name,
    #                 save_path=name, static_size=(None, *shape))


    dataset = load_dataset(name)
    X, Y = get_patch_dataset(dataset, 32, 10)
    print(X.shape, Y.shape)
    print(np.max(Y), np.min(X), np.max(X))
