import snorkel
from snorkel.labeling import labeling_function, LFApplier
from skimage.filters import threshold_li, threshold_otsu, threshold_sauvola, threshold_yen
from skimage.morphology import binary_dilation
from snorkel.labeling.model import LabelModel
from glob import glob
from pathlib import Path
from skimage import io
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops, label

def getLargestCC(segmentation):
    segmentation = binary_dilation(segmentation, np.ones((3,3,3)))
    labels = label(segmentation, connectivity=2)
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
    
    if log: print('Prepare arrays')

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
    LM = LabelModel(cardinality=2, verbose=True,device='cuda')    
    LM.fit(lab_arr, seed = 3333, log_freq=1, class_balance=[0.97, 0.03])

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


#########


if __name__ == '__main__':
    # calc(PATH_NEW = 'F:/Deep Learning/U-Net-3D-IRCAD/data/ircad_snorkel/antiga_3/*')
    # exit(1)
    dataset, filenames, original_images = load_dataset('F:/Deep Learning/Data/vesselness_ircad_ICPR/all/*', im_name='rorpo.nii')
    lfs = [li_thresholding, otsu_thresholding, yen_thresholding]
    labeling_applier(lfs, dataset, filenames, original_images, log=True)
    #applier = LFApplier(lfs=lfs)
    #L_train = applier.apply(dataset)
    #print(L_train)
    
