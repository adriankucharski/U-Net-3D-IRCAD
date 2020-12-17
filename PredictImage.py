import SimpleITK as sitk
from ImgDataFormatter import *
from functions import XPath
from skimage.transform import resize
from skimage.filters import threshold_li
class PredictImage():
    def __init__(self, model, imPath:str, imByIm:bool = True, allAxes:bool = False, normalization = False, staticSize = None, slicePerSlab:int = 16, stride:int = 1):
        self.orginalImg   = sitk.ReadImage(XPath(imPath))
        temp              = sitk.GetArrayFromImage(self.orginalImg)
        self.__allAxes    = allAxes
        self.__imByIm     = imByIm
        self.orginalShape = temp.shape
        self.im           = ImgDataFormatter(temp, normalization=normalization).getDataX(staticSize=staticSize)
        self.slicePerSlab = slicePerSlab
        self.stride       = stride
        self.model        = model
        self.__axial      = (0, 1, 2, 3, 4)
        self.__sagittal   = (0, 2, 3, 1, 4)
        self.__coronal    = (0, 3, 1, 2, 4)
        self.__axes       = [self.__axial, self.__sagittal, self.__coronal]
    
    def __validAxes(self, axes):
        if axes == self.__axial:
            return self.__axial
        if axes == self.__sagittal:
            return self.__coronal
        if axes == self.__coronal:
            return self.__sagittal
    
    def __predictImg(self, im, axes):
        im = np.transpose(im, axes)
        newIm = np.zeros(im.shape, dtype=np.float16)
        count = np.zeros(im.shape, dtype=np.float16)

        for slabID in range(0, im.shape[1] - self.slicePerSlab + 1, self.stride): 
            index = np.index_exp[:,slabID:slabID+self.slicePerSlab,:,:]
            slab = im[index]
            output = self.model.predict(slab, verbose = 0)
            newIm[index] += output[:,:,:,:,0:1]
            count[index] += 1
        
        count[np.where(count==0)] = 1
        newIm = newIm / count
        return np.transpose(newIm, self.__validAxes(axes))
    
    def __predictImByIm(self, im):
        shape = im.shape
        pred = np.ones(shape, dtype=np.float16)
        for idx in range(0, im.shape[1]):
            part = im[:, idx, :,:]
            pred[0, idx, :, :, :] = self.model.predict(part, batch_size = 1)
        return pred


    def predictImg(self):
        if self.__imByIm == True:
            return self.__predictImByIm(self.im)
        if self.__allAxes == True:
            newIm = np.zeros(self.im.shape, dtype=np.float16)
            for axes in self.__axes:
                newIm = newIm + self.__predictImg(self.im, axes)
            return newIm/3.0
        else:
            return self.__predictImg(self.im, self.__axial)

    def predictImgAndSave(self, path:str = 'test.nii'):
        print('Predicting image: ', path)
        newIm = self.predictImg()
        newIm = np.reshape(newIm, (newIm.shape[1], newIm.shape[2], newIm.shape[3]))
        newIm = np.array(newIm, dtype=np.float)
        newIm = resize(newIm, self.orginalShape)
        newIm = np.array(newIm*255.0, dtype=np.uint8)
        #newIm = (newIm > threshold_li(newIm)) * 255
        newIm = sitk.GetImageFromArray(newIm)
        newIm.CopyInformation(self.orginalImg)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(XPath(path))
        writer.Execute(newIm)
        