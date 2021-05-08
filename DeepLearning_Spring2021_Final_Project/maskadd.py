import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

########################################################################################################################
# Mask superposition
########################################################################################################################
head_path = "D:/SunXuhua/University/JHU_SP21_Baltimore/EN601.682 Deep Learning/project/DL_project/"
for i in range(670):
    folder_path = head_path + "data-science-bowl-2018/stage1_train/" + str(i) + "/masks/"
    mask_list = os.listdir(folder_path)
    # print(mask_list)
    length = len(mask_list)
    shape = mpimg.imread(folder_path + str(i) + "_0.png").shape
    img = np.zeros(shape)
    for mask_name in mask_list:
        mask_path = folder_path + mask_name
        temp = mpimg.imread(mask_path)
        img = img + temp
    save_path = folder_path + str(i) + ".png"
    plt.imsave(save_path, img, cmap='gray')
    # break  # Test for the first folder only

