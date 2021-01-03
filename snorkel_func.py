import snorkel
from snorkel.labeling import labeling_function, LFApplier
from skimage.filters import threshold_li, threshold_otsu, threshold_sauvola
from snorkel.labeling.model import LabelModel
from glob import glob
from pathlib import Path
from skimage import io
import numpy as np
import SimpleITK as sitk

def li_thresholding(im):
    threshold = threshold_li(im)
    return np.array(im > threshold, dtype=np.uint8)

def otsu_thresholding(im):
    threshold = threshold_otsu(im)
    return np.array(im > threshold, dtype=np.uint8)

def sauvola_thresholding(im):
    threshold = threshold_sauvola(im)
    return np.array(im > threshold, dtype=np.uint8)

def static_thresholding(im):
    threshold = 30
    return np.array(im > threshold, dtype=np.uint8)

class Image():
    def __init__(self, labels, shape):
        self.labels = labels
        self.shape = shape

def labeling_applier(lfs:list, dataset:list, filenames:list, save_perfix:str = 'data/ircad_snorkel', log:bool = False):
    """Base class for LF applier objects.

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
    
    Returns
    -------
        Labels
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
    

    if log: print('Training')
    LM = LabelModel(cardinality=2, verbose=True,device='cuda')    
    LM.fit(lab_arr, seed = 42, log_freq=1)

    if log: print('Predict')

    i = 0
    for image, name in zip(labeled_images, filenames):
        i += 1
        if log: print('Image: ' + str(i) + '/' + str(len(filenames)))

        im_flat = np.zeros(image.shape, dtype=np.uint8).flatten()
        
        p = LM.predict(image.labels)
        p[p > 0] = 255
        im_flat = np.reshape(p, image.shape).astype(np.uint8)

        new_im = sitk.GetImageFromArray(im_flat)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(Path(save_perfix) / (name + '.nii')))
        writer.Execute(new_im)


    


def load_dataset(path:str = 'F:/Deep Learning/Data/vesselness_ircad_ICPR/train/*', im_name:str = 'antiga.nii'):
    dataset = list()
    filenames = list()
    for dir_path in glob(str(Path(path))):
        im_path = Path(dir_path) / im_name
        im = io.imread(str(im_path))
        dataset.append(im)
        filenames.append(Path(dir_path).parts[-1] + '_' + im_name)
    return dataset, filenames



if __name__ == '__main__':
    dataset, filenames = load_dataset()
    lfs = [li_thresholding, otsu_thresholding, static_thresholding]
    labeling_applier(lfs, dataset, filenames, log=True)
    #applier = LFApplier(lfs=lfs)
    #L_train = applier.apply(dataset)
    #print(L_train)
    
