# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
import numpy as np


# define sone struct
RootPath = '/home1/Documents/Database/waymo/'  # root path
LeftFolder = 'left_imgs/'
RightFolder = 'right_imgs/'
RawDataType = '.jpg'
TestListPath = './Dataset/testlist_waymo.txt'


def OpenFile():
    if os.path.exists(TestListPath):
        os.remove(TestListPath)

    fd_test_list = open(TestListPath, 'a')

    return fd_test_list


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def ReadImg(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img


def GenList(left_imgs_folder_path, right_imgs_folder_path, fd_test_list):
    total = 0
    left_files = glob.glob(left_imgs_folder_path + '*' + RawDataType)
    for i in range(len(left_files)):
        name = os.path.basename(left_files[i])
        pos = name.find('left'+RawDataType)
        name = name[0:pos]

        left_img_path = left_files[i]
        right_img_path = right_imgs_folder_path + name + 'right'+RawDataType
        rawLeftPathisExists = os.path.exists(left_img_path)
        rawRightPathisExists = os.path.exists(right_img_path)

        if (not rawLeftPathisExists) or (not rawRightPathisExists):
            print "\"" + left_img_path + "\"" + "is not exist!!!"
            break

        img = ReadImg(left_img_path)
        print img.shape

        if img.shape[0] > 360 and img.shape[0]

        OutputData(fd_test_list, left_img_path)
        OutputData(fd_test_list, right_img_path)

        total = total + 1

    return total


if __name__ == '__main__':
    fd_test_list = OpenFile()
    total = GenList(RootPath + LeftFolder, RootPath+RightFolder, fd_test_list)
    print total
    # folderId = ConvertNumToChar(0)
    # folderNum = 0
    # num = 6
    # rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
    # print rawLeftPath
