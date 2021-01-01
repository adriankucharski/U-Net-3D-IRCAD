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
