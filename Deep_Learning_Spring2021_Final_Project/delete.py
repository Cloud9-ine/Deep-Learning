import os

########################################################################################################################
# Delete
########################################################################################################################
header_path = "D:/SunXuhua/University/JHU_SP21_Baltimore/EN601.682 Deep Learning/project/DL_project/"
for i in range(670):
    folder_path = header_path + "data-science-bowl-2018/stage1_train/" + str(i) +"/masks/"
    file_list = os.listdir(folder_path)
    # os.chdir(folder_path)
    for item in file_list:
        # print(item)
        if item == str(i) + ".png":
            continue
        else:
            os.remove(folder_path + item)
    # break  # Test for the first folder only
