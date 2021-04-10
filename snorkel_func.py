from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import snorkel
from skimage import io
from skimage.filters import (threshold_li, threshold_otsu, threshold_sauvola,
                             threshold_yen)
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from snorkel.labeling import LFApplier, labeling_function
from snorkel.labeling.model import LabelModel
import re
from dataset import io_load_image, io_save_image

def sorting(s): return int(re.findall(r'\d+', s)[-1])

def getLargestCC(segmentation):
    segmentation = binary_dilation(segmentation, np.ones((3,3,3)))
    labels:np.ndarray = label(segmentation, connectivity=2)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def ignore_black_backgroudn(im):
    im = im.flatten()
    return im[np.where(im != 0)]

def li_thresholding(im):
    im_mod = ignore_black_backgroudn(im)
    threshold = threshold_li(im_mod)
    return np.array(im > threshold, dtype=np.uint8)

def otsu_thresholding(im):
    im_mod = ignore_black_backgroudn(im)
    threshold = threshold_otsu(im_mod)
    return np.array(im > threshold, dtype=np.uint8)

def yen_thresholding(im):
    im_mod = ignore_black_backgroudn(im)
    threshold = threshold_yen(im_mod)
    return np.array(im > threshold, dtype=np.uint8)

def sauvola_thresholding(im):
    threshold = threshold_sauvola(im)
    return np.array(im > threshold, dtype=np.uint8)

def static_thresholding(im):
    threshold = 15
    return np.array(im > threshold, dtype=np.uint8)

class Image():
    def __init__(self, labels, shape):
        self.labels = labels
        self.shape = shape

def labeling_applier(lfs:list, dataset:list, filenames:list, original_images:list = None, save_perfix:str = 'data/ircad_snorkel', log:bool = False):
    """Function to generating label images.

    Parameters
    ----------
    lfs -
        LFs that this applier executes on examples

    dataset -
        List of numpy images 

    filenames - 
        list of filenames corresponding to dataset numpy images
    
    save_perfix - 
        folder save path

    log - 
        if true print status information
    """
    labeled_images = []

    size = 0
    for array in dataset:
        mul = 1
        for e in array.shape:
            mul *= e
        size += mul
    lab_arr = np.zeros((size, len(lfs)), dtype=np.uint8)
    
    if log: print('Prepare arrays', 'size:', size, 'bytes')

    index = 0
    for array in dataset:
        labeled_array = []
        for func in lfs:
            labeled_array.append(func(array).flatten())
        T = np.array(labeled_array).T
        lab_arr[index:index+T.shape[0], :] = T
        labeled_images.append(Image(T, array.shape))
        index += T.shape[0]
    
    #[[1 0 0 1], [0 1 0 1]]


    if log: print('Training')
    LM = LabelModel(cardinality=2, verbose=True, device='cuda')    
    LM.fit(lab_arr, seed = 3333, log_freq=1, class_balance=[0.965, 0.035])

    if log: print('Predict')

    iterator = zip(labeled_images, filenames, range(len(filenames)), range(len(filenames)))
    if original_images is not None:
        iterator = zip(labeled_images, filenames, range(len(filenames)), original_images)

    for array, name, idx, image in iterator:
        save_path = str(Path(save_perfix) / name)
        if log: print('Image: ' + str(idx + 1) + '/' + str(len(filenames)) + ' Save path: ' + save_path)

        im_flat = np.zeros(array.shape, dtype=np.uint8).flatten()
        
        #[[1 0 0 1], [0 1 0 1]]
        p = LM.predict(array.labels)

        #[[1] [0] [1]...]
        p = np.reshape(p, array.shape)
        p = getLargestCC(p)
        p[p > 0] = 255
        
        new_im = sitk.GetImageFromArray(np.array(p, dtype=np.uint8))
        if original_images is not None:
            new_im.CopyInformation(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(save_path)
        writer.Execute(new_im)


def load_dataset(path:str = 'F:/Deep Learning/Data/vesselness_ircad_ICPR/train/*', im_name:str = 'antiga.nii'):
    dataset = list()
    filenames = list()
    original_images = list()
    for dir_path in glob(str(Path(path))):
        im_path = Path(dir_path) / im_name
        img = sitk.ReadImage(str(im_path))
        
        im = sitk.GetArrayFromImage(img)
        dataset.append(im)
        original_images.append(img)
        filenames.append(Path(dir_path).parts[-1] + '_' + im_name)
    return dataset, filenames, original_images

def load_full_dataset(path:str = 'F:/Deep Learning/Data/vesselness_ircad_ICPR/all/*', im_names = ['antiga.nii', 'rorpo.nii']):
    dataset = list()
    filenames = list()
    original_images = list()
    for dir_path in glob(str(Path(path))):
        for im_name in im_names:
            im_path = Path(dir_path) / im_name
            img = sitk.ReadImage(str(im_path))
            
            im = sitk.GetArrayFromImage(img)
            dataset.append(im)
            original_images.append(img)
            filenames.append(Path(dir_path).parts[-1] + '_' + im_name)
    return dataset, filenames, original_images


#########

#################################################################################
#################################################################################
#################################################################################

def test():
    patient = 'maskedLiverIso.nii'
    filenames = ['antiga.nii', 'jerman.nii']
    labeling_functions = [li_thresholding, otsu_thresholding]
    path = 'F:/Deep Learning/Data/snorkel/*'

    dataset = []

    size = 0
    for dir_path in sorted(glob(str(Path(path))), key=sorting):
        ip: Path = Path(dir_path) / patient
        
        arr, im = io_load_image(str(ip))
        shape = arr.shape
        arr = arr.flatten()

        labels = []
        for fn in filenames:
            for func in labeling_functions:
                i = Path(dir_path) / fn
                array, image = io_load_image(str(i))
                array = func(array.flatten())
                size += array.shape[-1]
                labels.append(array)
        
        dataset.append([im, arr, labels, ip.parts[-2], shape])
    array_reshape = (size // len(filenames) // len(labeling_functions), len(filenames) * len(labeling_functions))
    print(array_reshape)

    lab:np.ndarray = np.zeros((size), dtype='float16').reshape(array_reshape)
    print(size, lab.shape)    
    s = 0
    for data in dataset:
        _, _, label, _, _  = data
        T: np.ndarray = np.array(label).T
        si = T.shape[0]
        lab[s:s+si, :] = T
        s += si

    LM: LabelModel = LabelModel(cardinality=2, verbose=True, device='cuda')    
    LM.fit(lab, seed = 12345, log_freq=1, n_epochs=100, class_balance=[0.985, 0.015])
    

    s = 0
    for data in dataset:
        im, arr, label, fn, shape  = data
        print(fn)
        T: np.ndarray = np.array(label).T
        p:np.ndarray = LM.predict(T)
        p = p.reshape(shape)
        p = getLargestCC(p)
        p[p > 0] = 255
        p = np.array(p, dtype='uint8')
        io_save_image('temp/' + fn + '.nii', p, im)


if __name__ == '__main__':
    test()
    exit()
    train_images = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


    # calc(PATH_NEW = 'F:/Deep Learning/U-Net-3D-IRCAD/data/ircad_snorkel/antiga_3/*')
    # exit(1)
    dataset, filenames, original_images = load_full_dataset('F:/Deep Learning/Data/vesselness_ircad_ICPR/train/*', im_names=['antiga.nii'])
    lfs = [li_thresholding, otsu_thresholding, yen_thresholding]
    labeling_applier(lfs, dataset, filenames, original_images, log=True)
    #applier = LFApplier(lfs=lfs)
    #L_train = applier.apply(dataset)
    #print(L_train)
    
