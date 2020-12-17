import numpy as np
from Resizer import Resizer


class ImgDataFormatter():
    def __init__(self, image, gt = None, normalization:bool = False):
        if gt is not None and image.shape != gt.shape:
            raise Exception('image shape and gt shape are not equal!')
        self.image          = image
        self.gt             = gt
        self.shape          = image.shape
        self.normalization  = normalization

    def getShape(self):
        return self.shape

    def getDataY(self, maxPools:int = 5, maxSize:int = 512, staticSize:list = None, label = (0, 1)):
        mask = Resizer(self.gt).getResizedIm(maxPools, maxSize, staticSize)
        depth, height, width = mask.shape
        mask = np.reshape(mask, (depth, height, width))
        newMask = np.zeros((depth, height, width, len(label)), dtype='float16')
        for i in range(0, len(label)):
            newMask[0:depth, 0:height, 0:width, i] = (mask == label[i])
        y = np.zeros((1, *newMask.shape), dtype=np.float16)
        y[:,:,:,:] = newMask
        return y

    def getDataX(self, maxPools:int = 5, maxSize:int = 512, staticSize:list = None):
        X = Resizer(self.image).getResizedIm(maxPools, maxSize, staticSize)
        if self.normalization == True:
            X = (X - np.mean(X)) / np.std(X)
        X = np.reshape(X, (*X.shape, 1))
        x = np.zeros((1, *X.shape), dtype=np.float16)
        x[:,:,:,:] = X
        return x

    def getDataXY(self, maxPools:int = 5, maxSize:int = 512, staticSize:list = None, label = (0, 1)):
        if self.gt is None:
            raise Exception('Cannot get data, gt is None.')
        X = self.getDataX(maxPools, maxSize, staticSize)
        Y = self.getDataY(maxPools, maxSize, staticSize, label)
        return X, Y
