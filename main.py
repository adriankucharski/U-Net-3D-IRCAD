from training import *
from predict import *


def train_3D():
    EPOCHES = 150
    VALIDATION_SPLIT = 0.2
    SLAB_PER_FILE = 128
    SLAB_SHAPE = (16, 224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 1
    DATASET_PATH = 'data/im_data.pickle'

    model_path = train_slab_unet3d(EPOCHES,
                                   VALIDATION_SPLIT,
                                   SLAB_PER_FILE,
                                   SLAB_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE)

    predict_images_slab('data/ircad_iso_111_test/*', str(model_path), static_size=STATIC_SIZE)


def train_2D():
    EPOCHES = 20
    VALIDATION_SPLIT = 0.2
    SLICE_PER_FILE = 128
    SLICE_SHAPE = (224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 16
    DATASET_PATH = 'data/im_data.pickle'

    # model_path = train_slice_unet2d(EPOCHES,
    #                                VALIDATION_SPLIT,
    #                                SLICE_PER_FILE,
    #                                SLICE_SHAPE,
    #                                DATASET_PATH,
    #                                BATCH_SIZE)

    predict_images_slice('data/ircad_iso_111_test/*', str('model/model_2D_02.01.2021_00-07-55.hdf5'), static_size=STATIC_SIZE)


if __name__ == '__main__':
    train_2D()