import pickle
import re
from glob import glob
from pathlib import Path

import numba
from skimage import io

from ImgDataFormatter import ImgDataFormatter



class PrepareDataset():
    def __init__(self, fromPath:str = 'C:/Users/adrian/Documents/CNN/ircad_iso_111/*', savePath:str = 'image_data.pickle', imName:str = 'patient.nii', gtName:str = 'liver.nii', normalization:bool = False):
        self.fromPath = fromPath
        self.savePath = savePath
        self.imName   = imName
        self.gtName   = gtName
        self.sorting  = lambda  s: int(re.findall(r'\d+', s)[-1])
        self.normalization = normalization

    def __loadImgDataset(self, maxPools:int = 5, maxSize:int = 512, staticSize:list = None, label = (0, 1)):
        imData = []
        for dirPath in sorted(glob(str(Path(self.fromPath))), key=self.sorting):
            imPath = Path(dirPath) / self.imName
            gtPath = Path(dirPath) / self.gtName
            X, Y = ImgDataFormatter(io.imread(str(imPath)), io.imread(str(gtPath)), self.normalization).getDataXY(maxPools, maxSize, staticSize, label)
            imData.append((X, Y))
        return imData
    
    def saveImgDataset(self, maxPools:int = 5, maxSize:int = 512, staticSize:list = None, label = (0, 1)):
        imData = self.__loadImgDataset(maxPools, maxSize, staticSize, label)
        with open(str(Path(self.savePath)), 'wb') as file:
            pickle.dump(imData, file)
    
    @staticmethod
    def loadImgDataset(path:str = 'image_data.pickle'):
        imData = None
        with open(str(Path(path)), 'rb') as file:
            imData = pickle.load(file)
        return imData

if __name__ == '__main__':
    PrepareDataset(
        fromPath      = 'data/ircad_iso_111/*', 
        savePath      = 'data/image_data_224x224.pickle', 
        imName        = 'patientIso.nii', 
        gtName        = 'vesselsIso.nii', 
        normalization = True
    ).saveImgDataset(
        staticSize  = (None, 224, 224), 
        label       = [1]
    )

