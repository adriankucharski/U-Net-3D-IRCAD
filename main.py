from training import *
from predict import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_3D():
    EPOCHES = 75
    VALIDATION_SPLIT = 0.2
    SLAB_PER_FILE = 145
    SLAB_SHAPE = (5, 224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 4
    DATASET_PATH = 'data/im_liver_16train_4test_notanh.pickle'
    NETWORK = experimental_network_3D
    SLICE_PER_SLAB = SLAB_SHAPE[0]

    model_path = train_slab_unet3d(EPOCHES,
                                   VALIDATION_SPLIT,
                                   SLAB_PER_FILE,
                                   SLAB_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE,
                                   NETWORK)

    predict_images_slab('data/ircad_iso_111_test/*', str(model_path), static_size=STATIC_SIZE, slice_per_slab=SLICE_PER_SLAB)


def train_2D():
    EPOCHES = 300
    VALIDATION_SPLIT = 0.2
    SLICE_PER_FILE = 145
    SLICE_SHAPE = (224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 16
    DATASET_PATH = 'data/im_liver_16train_4test_notanh.pickle'
    NETWORK = UNet2D
    DATASET_TEST = 'data/ircad_test/*'

    model_path = train_slice_unet2d(EPOCHES,
                                   VALIDATION_SPLIT,
                                   SLICE_PER_FILE,
                                   SLICE_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE,
                                   NETWORK)

    predict_images_slice(DATASET_TEST, str(model_path), static_size=STATIC_SIZE)

def train_generator_2D():
    EPOCHES = 200
    VALIDATION_SPLIT = 0.2
    SLICE_PER_FILE = 128
    SLICE_SHAPE = (224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 8
    DATASET_PATH = 'data/im_vessels_data_notanh.pickle'
    NETWORK = UNet2D
    STEPS_PER_EPOCH = 100

    model_path = train_slice_generator_unet2d(EPOCHES,
                                   SLICE_PER_FILE,
                                   VALIDATION_SPLIT,
                                   SLICE_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE,
                                   NETWORK,
                                   STEPS_PER_EPOCH)

    predict_images_slice('data/ircad_iso_111_test/*', str(model_path), static_size=STATIC_SIZE)


if __name__ == '__main__':
    train_2D()