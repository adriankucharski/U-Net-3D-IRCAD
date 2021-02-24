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
from snorkel_func import getLargestCC
from scipy.ndimage import filters
from skimage.morphology import binary_erosion
from dataset import *
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
    
    
    intersection = np.sum(pred * true) * 2.0
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

def threshold_result(im):
    elem = np.ones((3,3,3))/255.
    im = filters.convolve(im, elem) 
    print(im.max())
    im = im > threshold_li(im)
    return im
    #im = filter()

from skimage.filters import threshold_li, threshold_otsu, threshold_sauvola, threshold_yen
def calculate_result(path = 'data/ircad_test', path_pred = 'predicted/*.nii', image_name = 'vesselsIso.nii', mask = None):
    print(path_pred)
    for dir_new in glob(path_pred):
        im_name:str = Path(dir_new).parts[-1]
        folder = im_name.split('_')[0]
        dir_old = Path(path) / folder / image_name

        mask_path = None if mask is None else Path(path) / folder / mask
        try:
            im_1 = io.imread(dir_new)
            im_2 = io.imread(dir_old)
            
            if mask_path is not None:
                im_3 = io.imread(mask_path)
                im_2 = im_2 * im_3

            im_1 = im_1 > threshold_li(im_1[np.where(im_1 > 0)])
            im_2 = im_2 > 0

            #im_1=binary_erosion(im_1, np.ones((3,3,3)))
            #im_1 = getLargestCC(im_1)
            print(dir_new, calculate_dice(im_1, im_2))
        except:
            pass

def threshold_results_and_save(path_pred = 'predicted/*.nii', path_save = 'predicted/tresholded'):
    for dir_new in glob(path_pred):
        im, data = io_load_image(dir_new)
        im = np.array((im > threshold_li(im[np.where(im > 0)])) * 255, dtype=np.uint8)
        
        im_name:str = Path(dir_new).parts[-1]
        new_dir = Path(path_save) / im_name 
        io_save_image(str(new_dir), im, data)



if __name__ == '__main__':
    # threshold_results_and_save('predicted/snorkel_2D_res/*.nii')

    # exit()
    
    print('Liver')
    calculate_result(path="data/ircad_snorkel/antiga098-002/test", path_pred='predicted/snorkel_3D_res/*.nii', image_name='antiga.nii')
    print('-' * 50)
    print('Vessels')
    #calculate_result('data/ircad_iso_111_test', path_pred='predicted/Unet3D_old_res/*.nii',  image_name='vesselsIso.nii')
