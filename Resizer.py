import numpy as np
from skimage.transform import resize
"""
Klasa {@code Resizer} zmienia rozmiar obrazu w taki sposób, 
aby można było wykonać na nim maxPools operacji MaxPooling (2D/3D)
"""
class Resizer():
    """
    Class Resizer is used for change input image size.
    
    Methods
    -------
    - `getNewSize`
    - `getResizedIm`
    """
    def __init__(self, image):
        self.image      = image

    def getNewSize(self, maxPools:int = 5, sizeMax:int = 512) -> tuple:
        """
        Method `getNewSize` returns new image size `(tuple)`
        
        Parameters
        ----------
        `maxPools:int`
        - With `maxPools = 5` we can use `MaxPooling(2)` operation on image `5` times
        - Value should be >= 1
        `sizeMax:int`
        - Maximum size of each dimension of resized image
        - Value should be >= 8
        """
        assert maxPools >= 1 and sizeMax >= 8
        size = self.image.shape
        newSize = []
        validSize  = np.array(np.arange(2**maxPools, sizeMax, 2**maxPools))

        for k in size:
            idx = np.argmin(np.abs(validSize - k))
            newSize.append(validSize[idx])

        return tuple(newSize)
    
    def getResizedIm(self, maxPools:int = 5, sizeMax:int = 512, staticSize:list = None, antiAliasing = False):
        """
        Method getResizedIm returns resized image 
        with new size calculated by getNewSize method, or given in staticSize argument.

        Parameters
        ----------
        `maxPools:int`
        - With `maxPools = 5` we can use `MaxPooling(2)` operation on image `5` times
        - Value should be >= 1
        `sizeMax:int`
        - Maximum size of each dimension of resized image
        - Value should be >= 8

        `staticSize : list`

        - Resize image to defined size
        - Dimension have to be the same as image
        >>> staticSize = None - image will be resized to size returned by getNewSize method

        >>> staticSize = (128, 256, 256) - image will be resized to (128, 256, 256)

        >>> staticSize = (None, 256, 256) - image will be resized to (im.shape[0], 256, 256)

        `antiAliasing : bool`
        - Resize image with anti aliasing
        """
        assert len(staticSize) == len(self.image.shape)

        newSize = list(self.image.shape)
        if staticSize == None:
            newSize = self.getNewSize()
        else:
            for i in range(len(staticSize)):
                newSize[i] = newSize[i] if staticSize[i] == None else staticSize[i]
        return np.array(resize(self.image, newSize, anti_aliasing=antiAliasing), dtype=np.float16)
