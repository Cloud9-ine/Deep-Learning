import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io

########################################################################################################################
# Demo
########################################################################################################################

head_path = "D:/SunXuhua/University/JHU_SP21_Baltimore/EN601.682 Deep Learning/project/DL_project/"
folder_path = head_path + "data-science-bowl-2018/stage1_train/0/masks/"

img = io.imread(folder_path + "0.png")
for i in range(img.shape[0]):
    print(img[i])
