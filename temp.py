import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Accuracy, MeanIoU, Precision, Recall
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv3D, Dropout, Input, LeakyReLU,
                                     MaxPooling2D, MaxPooling3D, Softmax,
                                     UpSampling2D, UpSampling3D, concatenate)
from tensorflow.keras.activations import sigmoid, softmax, tanh
from tensorflow.keras import backend as K
import tensorflow as tf
###################################################
#                   Metrics
###################################################
def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = 2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth/2.0
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return intersection / sums
 
def gen_crossvalidation_array(num_of_el: int):
    a = list(range(num_of_el))
    a = [x+1 for x in a]
    cross = []
    for i in range(len(a)):
        test = i + 1
        train = list(a)
        train.remove(test)
        cross.append([train, test])
    return cross


from History import History
if __name__ == '__main__':
    h = History().load_history('history/cross/history_.hist_0')
    h.plot_all()

    
