import pickle
import random
import re
from glob import glob
from pathlib import Path

import numba
import numpy as np
from skimage import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PrepareDataset import PrepareDataset
import SimpleITK as sitk
from skimage.transform import resize
from datetime import datetime
import tensorflow as tf

from model import UNet3D
from tensorflow.keras.callbacks import ModelCheckpoint
from History import History



def XPath(path: str):
    return str(Path(path))

# ************************************************************
#                       GENERATOR
# ************************************************************


def SimpleImageGenerator(picklePath: str = 'image_data.pickle', slicePerSlab: int = 16, slabPerFile: int = 16):
    print("Loading data...")
    imData = None
    with open(str(Path(picklePath)), 'rb') as file:
        imData = pickle.load(file)

    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    while True:
        filesID = list(range(0, numOfFiles))
        random.shuffle(filesID)
        while len(filesID) > 0:
            i = 0
            IDs = []
            x, y = imData[filesID.pop()]
            while i < slabPerFile:
                slabID = int(np.random.randint(
                    0, x.shape[1] - slicePerSlab - 1))

                if slabID in IDs:
                    continue
                i += 1
                IDs.append(slabID)
                index = np.index_exp[:, slabID:slabID+slicePerSlab, :, :]
                yield x[index], y[index]


def SimpleImageGenerator2D(picklePath: str = 'image_data.pickle', imagesPerFile: int = 128):
    print("Loading data...")
    imData = None
    with open(str(Path(picklePath)), 'rb') as file:
        imData = pickle.load(file)

    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    while True:
        for ID in range(0, numOfFiles):
            i = 0
            IDs = []
            x, y = imData[ID]
            while i < imagesPerFile:
                slabID = int(np.random.randint(0, x.shape[1]))

                if slabID in IDs:
                    continue
                i += 1
                IDs.append(slabID)
                index = np.index_exp[0, slabID, :, :]
                yield x[index], y[index]


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


# ************************************************************
#                       DATASET
# ************************************************************


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

    for i in range(0, len(label)):
        new_gt[0:depth, 0:height, 0:width, i] = (gt == label[i])

    reshaped = np.zeros((1, *new_gt.shape), dtype=np.float16)
    reshaped[:, :, :, :] = new_gt
    return reshaped


def load_pair_images(im_path, gt_path) -> list:
    X = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(im_path))))
    Y = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(gt_path))))
    return [X, Y]


def prepare_dataset(from_path='data/ircad_iso_111/*', im_name='patient.nii', gt_name='liver.nii', save_path='data/im_data.pickle', static_size=None):
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
        Y = preproces_im(Y)
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

# ************************************************************
#                       LOGS
# ************************************************************


def im_info(path: str = '', save_path: str = 'ircad_iso_111.csv'):
    def sorting(s): return int(re.findall(r'\d+', s)[-1])
    foldersPath = Path(path)

    csv = open(XPath(save_path), 'w')
    csv.writelines('image, depth, height, width\n')
    for folder in sorted(glob(str(foldersPath)), key=sorting):
        imPath = Path(folder)
        im = io.imread(imPath / 'patientIso.nii')
        z, y, x = im.shape
        print(im.shape)
        csv.writelines('%s, %d, %d, %d\n' % (imPath.parts[-1], z, y, x))
    csv.close()

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


# ************************************************************
#                       TRAINING
# ************************************************************

EPOCHES = 150
VALIDATION_SPLIT = 0.2
SLAB_PER_FILE = 128
SLAB_SHAPE = (16, 224, 224, 1)


def train_slab_unet3d():
    dataset = load_dataset('data/new_image_data_iso.pickle')
    num_of_images = len(dataset)
    slabs_number = num_of_images * SLAB_PER_FILE

    X, Y = generate_slab_dataset(dataset, SLAB_PER_FILE, SLAB_SHAPE)
    model = UNet3D(SLAB_SHAPE)


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


        checkpointer = ModelCheckpoint(
            str(model_save), 'val_loss', 2, True, mode='auto')
        history = model.fit(X, Y, verbose=1, epochs=EPOCHES, batch_size=1,
                            validation_split=VALIDATION_SPLIT, callbacks=[checkpointer])
        
        hist = History(history)
        hist.save_history(history_save)

    predict_images_slab('data/ircad_iso_111_test/*')


# dopisać training


if __name__ == '__main__':
    predict_images_slab(
        imgs_path='data/ircad_iso_111_test/*',
        model_path='model/20201114_2235_model_iso_2.hdf5',
    )
    # im_info('ircad_iso_111_full/*', 'temp/ircad_iso_111.csv')

    # prepare_dataset(
    #     from_path='data/ircad_iso_111/*',
    #     im_name='patientIso.nii',
    #     gt_name='vesselsIso.nii',
    #     save_path='data/new_image_data_iso.pickle',
    #     static_size=(None, 224, 224)
    # )

    # data = load_dataset('data/new_image_data_iso.pickle')
    # print(len(data))
