import os
from glob import glob
from sys import getsizeof
import gc


from scipy.spatial import distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

from functions import *
from History import *
from model import SegNet3D, UNet3D, UNet2D
from model import dice_coef_loss, tversky_loss, iou_loss
from PredictImage import *




def predictImgs(imgsPath:str = 'ircad_iso_111/*', imName:str = 'patient.nii', modelPath:str = 'model/model.hdf5', allAxes:bool = False, fullImages:bool = False, slicePerSlab = 8):
    model = tf.keras.models.load_model(XPath(modelPath), custom_objects={'dice_coef_loss': dice_coef_loss})
    for dirPath in glob(XPath(imgsPath)):
        suffix = Path(dirPath).parts[-1]
        imPath = Path(dirPath) / imName
        pathSave = Path('predicted') / (suffix + '_patient.nii')
        PredictImage(model, imPath, 
            imByIm=False,
            allAxes=allAxes, 
            stride=2, 
            #fullImage=fullImages, 
            slicePerSlab=slicePerSlab, 
            normalization=True,
            staticSize=(None, 224, 224)
        ).predictImgAndSave(str(pathSave))

def calc_dice(im1, im2):
    return 1 - distance.dice(im1.ravel(), im2.ravel())

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

import skimage
from skimage.filters import threshold_li
def calc_results():
    path = './predicted/'
    for i in [1, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]:
        pathGT = './predicted/' + str(i) + '.nii'
        pathIM = './predicted/3Dircadb1.' + str(i) + '_patient.nii'
        
        gt = io.imread(pathGT)
        im = io.imread(pathIM)
        im = im > threshold_li(im)

        print(str(i) + ": " + str(dice(im, gt)))
    exit()
    
def predict_2D(model_path, images_path:str, imName:str = 'patientIso.nii', path_save = Path('predicted'), suffixSave = '_patient.nii'):
    model = tf.keras.models.load_model(XPath(model_path), custom_objects={'loss': iou_loss})
    for dirPath in glob(XPath(images_path)):
        suffix = Path(dirPath).parts[-1]
        imPath = Path(dirPath) / imName
        p_save = path_save / (suffix + suffixSave)

        PredictImage(
            model = model,
            imPath = imPath,
            imByIm = True,
            staticSize = (None, 224, 224),
            normalization = True
        ).predictImgAndSave(path = p_save)

def training_Unet2D():
    model_save = 'model/20201120_1711_model_unet_2d_2.hdf5'
    history_save = 'history/20201120_1711_model_unet_2d_2.pickle'
    pickle_path = 'data/image_data_224x224.pickle'

    files = 15
    images = 160
    epochs = 300
    
    ###############
    GEN = SimpleImageGenerator2D(picklePath='data/image_data_224x224.pickle', imagesPerFile=images)
    X = np.empty((images * files, 224, 224, 1), dtype=np.float16)
    Y = np.empty((images * files, 224, 224, 1), dtype=np.float16)
    
    for idx in range(images * files):
        x, y = next(GEN)
        X[idx,:,:,:] = x
        Y[idx,:,:,:] = y

    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]
    
    ###############
    model = UNet2D((224, 224, 1))
    checkpointer = ModelCheckpoint(model_save, 'val_loss', 2, True, mode='auto')
    history = model.fit(X, Y, verbose=1, epochs=epochs, batch_size=8, validation_split=0.2, callbacks=[checkpointer], use_multiprocessing=True)
    hist = History(history)
    hist.save_history(history_save)
    hist.save_all('history/plots/')

    #####################
    predict_2D(model_save, 'data/ircad_iso_111_test/*')
    calc_results()

    exit(0)
    


if __name__ == '__main__':


    weights_path = 'model/20201119_1248_model_bc_split.hdf5'
    model_save = 'trained_model.hdf5'
    history_save = 'history.pickle'
    epochs = 150
    slabSize = 16
    files = 15
    slabPerFile = 128
    predictOnly = False

    gen = SimpleImageGenerator(
        picklePath='image_data_224x224.pickle', 
        slicePerSlab=slabSize, 
        slabPerFile=slabPerFile
    )

    SLABS = slabPerFile * files

    X = np.empty((SLABS, 16, 224, 224, 1), dtype=np.float16)
    Y = np.empty((SLABS, 16, 224, 224, 1), dtype=np.float16)

    for idx in range(SLABS):
        x, y = next(gen)
        X[idx,:,:,:,:] = x
        Y[idx,:,:,:,:] = y

    gen.close()
    gc.collect()
    print('X = ' + str(int(getsizeof(X) / 2**20)) + ' MB')
    print('Y = ' + str(int(getsizeof(Y) / 2**20)) + ' MB')
    
    if predictOnly == False:
        model = UNet3D((slabSize, 224, 224, 1))
        checkpointer = ModelCheckpoint(model_save, 'val_loss', 2, True, mode='auto')
        history = model.fit(X, Y, verbose=1, epochs=epochs, batch_size=1, validation_split=0.2, callbacks=[checkpointer], use_multiprocessing=True, workers=2)

        #hist = History(history)
        #hist.save_history(history_save)
        #hist.save_plot_history('history/plots/')

    predictImgs(imgsPath = 'data/ircad_iso_111_test/*', modelPath='trained_model.hdf5', imName='patientIso.nii', allAxes=False, slicePerSlab=slabSize)
    calc_results()

"""
alfa = 0.05, beta=0.05
16: 0.6939153901411459
17: 0.5914304528644861
18: 0.5804069675951145
19: 0.6400342176588827
20: 0.534273974164330

alfa = 0.15, beta=0.20
16: 0.5655243108879886
17: 0.3845109587487991
18: 0.4055981542472897
19: 0.5729255823636911
20: 0.4038453950810645
trevsky alfa = beta = 0.1
16: 0.6341668860226375
17: 0.4925626017283873
18: 0.4442464425168374
19: 0.6221860064360968
20: 0.4743804670771989
trevsky alfa = beta = 0.01
16: 0.5707497340093219
17: 0.49380167271373726
18: 0.43943718458607667
19: 0.6122343920153357
20: 0.4450055979144704

16: 0.721398456993719
17: 0.6298613933942874
18: 0.6241567230632236
19: 0.6571566073464946
20: 0.5930041218264342

obraz do obrazu
16: 0.6914317119774922
17: 0.6940638516539985
18: 0.6195134476361309
19: 0.716969426188332
20: 0.5952706326937124

16: 0.7291939062268895
17: 0.7316478907645404
18: 0.709714059953578
19: 0.715986961657865
20: 0.6076953107500112

16: 0.6950709654463509
17: 0.7061598104673702
18: 0.5969286005505636
19: 0.621237645454988
20: 0.5497635656642632

16: 0.7167661625315298
17: 0.7447058595591607
18: 0.7254251634579674
19: 0.7208122648760164
20: 0.6490804207668815

16: 0.709874123437432
17: 0.7298675960019314
18: 0.7374935303671182
19: 0.7267719706948542
20: 0.6389427312775331

16: 0.696
17: 0.648
18: 0.610
19: 0.699
20: 0.547
"""