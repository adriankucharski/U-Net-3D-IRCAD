from training import *
from predict import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_3D():
    EPOCHES = 25
    VALIDATION_SPLIT = 0.2
    SLAB_PER_FILE = 145
    SLAB_SHAPE = (16, 224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 1
    DATASET_PATH = 'data/im_antiga098_002_masked_16train_4test.pickle'
    NETWORK = UNet3D
    SLICE_PER_SLAB = SLAB_SHAPE[0]
    DATASET_TEST = 'data/ircad_snorkel/antiga098-002/test/*'

    model_path = train_slab_unet3d(EPOCHES,
                                   VALIDATION_SPLIT,
                                   SLAB_PER_FILE,
                                   SLAB_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE,
                                   NETWORK)

    predict_images_slab(DATASET_TEST, str(model_path), static_size=STATIC_SIZE, slice_per_slab=SLICE_PER_SLAB)


def train_patch_3D():
    EPOCHES = 50
    VALIDATION_SPLIT = 0.2
    PATCH_PER_FILE = 20000
    PATCH_SIZE = 16
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 64
    DATASET_PATH = 'data/im_antiga098_002_masked_16train_4test.pickle'
    NETWORK = UNet3DPatch
    
    model_path = train_patch_unet3d(
        EPOCHES,
        VALIDATION_SPLIT,
        PATCH_PER_FILE,
        PATCH_SIZE,
        DATASET_PATH,
        BATCH_SIZE,
        NETWORK
    )

def train_2D():
    EPOCHES = 300
    VALIDATION_SPLIT = 0.2
    SLICE_PER_FILE = -1
    SLICE_SHAPE = (224, 224, 1)
    STATIC_SIZE = (None, 224, 224)
    BATCH_SIZE = 16
    DATASET_PATH = 'data/im_antiga098_002_masked_16train_4test.pickle'
    NETWORK = UNet2D
    DATASET_TEST = 'data/ircad_snorkel/antiga098-002/test/*'

    model_path = train_slice_unet2d(EPOCHES,
                                   VALIDATION_SPLIT,
                                   SLICE_PER_FILE,
                                   SLICE_SHAPE,
                                   DATASET_PATH,
                                   BATCH_SIZE,
                                   NETWORK)

    predict_images_slice(DATASET_TEST, str(model_path), im_name='patientIso.nii', static_size=STATIC_SIZE)

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
    train_patch_3D()