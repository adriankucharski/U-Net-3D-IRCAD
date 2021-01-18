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
from History import History


def XPath(path: str):
    return str(Path(path))

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

def calculate_dice(a:np.ndarray, b:np.ndarray):
    pred = a.flatten()
    true = b.flatten()
    
    
    intersection = np.sum(pred[true > 0]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def calc(PATH = 'F:/Deep Learning/U-Net-3D-IRCAD/data/ircad_iso_111',
 PATH_NEW = 'F:/Deep Learning/U-Net-3D-IRCAD/data/ircad_snorkel/rorpo/*' ):

    for dir_new in glob(PATH_NEW):
        im_name:str = Path(dir_new).parts[-1]

        folder = im_name.split('_')[0]
        filename = 'vesselsIso.nii'
        dir_old = Path(PATH) / folder / filename

        im_1 = io.imread(dir_new)
        im_2 = io.imread(dir_old)

        print(np.max(im_2))
        im_1[im_1 > 0] = 1
        im_2[im_2 > 0] = 1
        print(im_name, calculate_dice(im_1, im_2))

def calculate_result(path = 'data/ircad_iso_111_test', path_pred = 'predicted/Unet2D_experimental/*.nii', image_name = 'liverMaskIso.nii'):
    for dir_new in glob(path_pred):
        im_name:str = Path(dir_new).parts[-1]
        folder = im_name.split('_')[0]
        dir_old = Path(path) / folder / image_name
        try:
            im_1 = io.imread(dir_new)
            im_2 = io.imread(dir_old)

            im_1[im_1 > 0] = 1
            im_2[im_2 > 0] = 1
            print(im_name, calculate_dice(im_1, im_2))
        except:
            pass
if __name__ == '__main__':
    calculate_result()
