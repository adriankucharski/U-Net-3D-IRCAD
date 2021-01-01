from training import *
from predict import *

if __name__ == '__main__':
    EPOCHES = 150,
    VALIDATION_SPLIT = 0.2
    SLAB_PER_FILE = 128
    SLAB_SHAPE = (16, 224, 224, 1)

    model_path = train_slab_unet3d(EPOCHES=5,
                                   VALIDATION_SPLIT=0.2,
                                   SLAB_PER_FILE=128,
                                   SLAB_SHAPE=(16, 224, 224, 1))
    
    predict_images_slab('data/ircad_iso_111_test/*', str(model_path))
