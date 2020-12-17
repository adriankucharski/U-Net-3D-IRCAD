import pickle
import random
import re
from glob import glob
from pathlib import Path

import numba
import numpy as np
from skimage import io

from PrepareDataset import PrepareDataset


def XPath(path: str):
    return str(Path(path))




def SimpleImageGenerator(picklePath: str = 'image_data.pickle', slicePerSlab: int = 16, slabPerFile: int = 16):
    print("Loading data...")
    imData = None
    with open(str(Path(picklePath)), 'rb') as file:
        imData = pickle.load(file)

    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    while True:
        filesID = list(range(0, numOfFiles))
        random.shuffle(filesID)
        while len(filesID) > 0:
            i = 0
            IDs = []
            x, y = imData[filesID.pop()]
            while i < slabPerFile:
                slabID = int(np.random.randint(
                    0, x.shape[1] - slicePerSlab - 1))
                    
                if slabID in IDs:
                    continue
                i += 1
                IDs.append(slabID)
                index = np.index_exp[:, slabID:slabID+slicePerSlab, :, :]
                yield x[index], y[index]

def SimpleImageGenerator2D(picklePath: str = 'image_data.pickle', imagesPerFile:int = 128):
    print("Loading data...")
    imData = None
    with open(str(Path(picklePath)), 'rb') as file:
        imData = pickle.load(file)

    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    while True:
        for ID in range(0, numOfFiles):
            i = 0
            IDs = []
            x, y = imData[ID]
            while i < imagesPerFile:
                slabID = int(np.random.randint(0, x.shape[1]))
        
                if slabID in IDs: 
                    continue
                i += 1
                IDs.append(slabID)
                index = np.index_exp[0,slabID,:,:]  
                yield x[index], y[index]
            


def ImageGenerator(picklePath: str = 'image_data.pickle', slicePerSlab: int = 16, slabPerFile: int = 16, threeViews: bool = False, dataArg=None):
    print("Loading data...")

    imData = dataArg
    if imData is None:
        with open(str(Path(picklePath)), 'rb') as file:
            imData = pickle.load(file)

    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    if threeViews == False:
        while True:
            filesID = list(range(0, numOfFiles))
            random.shuffle(filesID)

            while len(filesID) > 0:
                x, y = imData[filesID.pop()]
                for _ in range(slabPerFile):
                    slabID = int(np.random.randint(
                        0, x.shape[1] - slicePerSlab - 1))
                    index = np.index_exp[:, slabID:slabID+slicePerSlab, :, :]
                    yield x[index], y[index]
    else:
        axial = (0, 1, 2, 3, 4)
        sagittal = (0, 2, 3, 1, 4)
        coronal = (0, 3, 1, 2, 4)
        views = [axial, coronal, sagittal]

        while True:
            filesID = list(range(0, numOfFiles))
            random.shuffle(filesID)

            while len(filesID) > 0:
                dataID = filesID.pop()
                random.shuffle(views)
                for view in views:
                    x, y = imData[dataID]
                    x = np.transpose(x, view)
                    y = np.transpose(y, view)
                    for _ in range(slabPerFile):
                        slabID = int(np.random.randint(
                            0, x.shape[1] - slicePerSlab - 1))
                        index = np.index_exp[:,
                                             slabID:slabID+slicePerSlab, :, :]
                        yield x[index], y[index]


def FullImageGenerator(picklePath: str = 'image_data.pickle'):
    print("Loading data...")
    imData = PrepareDataset().loadImgDataset(XPath(picklePath))
    numOfFiles = len(imData)
    print("Data loaded successful. Number of files %d" % numOfFiles)

    while True:
        indexes = list(range(0, numOfFiles))
        random.shuffle(indexes)
        for index in indexes:
            x, y = imData[index]
            yield x, y


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


if __name__ == '__main__':
    im_info('ircad_iso_111_full/*', 'temp/ircad_iso_111.csv')
