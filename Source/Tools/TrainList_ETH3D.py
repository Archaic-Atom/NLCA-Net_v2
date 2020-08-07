# -*- coding: utf-8 -*-

import os
import glob


from PIL import Image
import numpy as np


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def ReadImg(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img


TrainListPath = './Dataset/trainlist_ETH3D.txt'
DispLabelListPath = './Dataset/labellist_disp_ETH3D.txt'

ValTrainListPath = './Dataset/testlist_ETH3D.txt'
# ValDispLabelListPath = './Dataset/test_label_disp_list_ETH3D.txt'
# ValClsLabelListPath = './Dataset/test_label_cls_list_CityScape.txt'


RootPath = '/home1/Documents/Database/ETH3D_Stereo/'


# cls_folder_list = ['leftImg8bit/', 'rightImg8bit/', 'gtCoarse/', 'disparity/']


train_folder_list = ['TRAIN/delivery_area_1l', 'TRAIN/delivery_area_1s',
                     'TRAIN/delivery_area_2l', 'TRAIN/delivery_area_2s',
                     'TRAIN/delivery_area_3l', 'TRAIN/delivery_area_3s',
                     'TRAIN/electro_1l', 'TRAIN/electro_1s',
                     'TRAIN/electro_2l', 'TRAIN/electro_2s',
                     'TRAIN/electro_3l', 'TRAIN/electro_3s',
                     'TRAIN/facade_1s', 'TRAIN/forest_1s',
                     'TRAIN/forest_2s', 'TRAIN/playground_1l',
                     'TRAIN/playground_1s', 'TRAIN/playground_2l',
                     'TRAIN/playground_2s', 'TRAIN/playground_3l',
                     'TRAIN/playground_3s', 'TRAIN/terrace_1s',
                     'TRAIN/terrace_2s', 'TRAIN/terrains_1l',
                     'TRAIN/terrains_1s', 'TRAIN/terrains_2l',
                     'TRAIN/terrains_2s']

val_folder_list = ['TEST/lakeside_1l', 'TEST/lakeside_1s', 'TEST/sand_box_1l',
                   'TEST/sand_box_1s', 'TEST/storage_room_1l',
                   'TEST/storage_room_1s', 'TEST/storage_room_2_1l',
                   'TEST/storage_room_2_1s', 'TEST/storage_room_2_2l',
                   'TEST/storage_room_2_2s', 'TEST/storage_room_2l',
                   'TEST/storage_room_2s', 'TEST/storage_room_3l',
                   'TEST/storage_room_3s', 'TEST/tunnel_1l',
                   'TEST/tunnel_2l', 'TEST/tunnel_2s', 'TEST/tunnel_1s',
                   'TEST/tunnel_3l', 'TEST/tunnel_3s']


if os.path.exists(TrainListPath):
    os.remove(TrainListPath)

if os.path.exists(DispLabelListPath):
    os.remove(DispLabelListPath)

if os.path.exists(ValTrainListPath):
    os.remove(ValTrainListPath)

fd_train_list = open(TrainListPath, 'a')
fd_disp_label_list = open(DispLabelListPath, 'a')
fd_val_train_list = open(ValTrainListPath, 'a')

#fd_train_list = None
#fd_disp_label_list = None
#fd_val_train_list = None

img_h = 9999999
img_w = 9999999

for i in range(len(train_folder_list)):
    path = RootPath + train_folder_list[i]

    path_0 = path + '/im0.png'
    path_1 = path + '/im1.png'
    path_2 = path + '/disp0GT.pfm'

    exist_0 = os.path.exists(path_0)
    exist_1 = os.path.exists(path_1)
    exist_2 = os.path.exists(path_2)

    if (not exist_0) or \
        (not exist_1) or \
            (not exist_2):
        print "'" + path_0 + "' : is not existed!"
        print "'" + path_1 + "' : is not existed!"
        print "'" + path_2 + "' : is not existed!"
        print '***************'
        break

    img = ReadImg(path_0)
    print img.shape

    if img.shape[0] < img_h:
        img_h = img.shape[0]

    if img.shape[1] < img_w:
        img_w = img.shape[1]

    OutputData(fd_train_list, path_0)
    OutputData(fd_train_list, path_1)
    OutputData(fd_disp_label_list, path_2)
    print "Finish: " + train_folder_list[i]


print img_h, img_w

for i in range(len(val_folder_list)):
    path = RootPath + val_folder_list[i]

    path_0 = path + '/im0.png'
    path_1 = path + '/im1.png'

    exist_0 = os.path.exists(path_0)
    exist_1 = os.path.exists(path_1)

    if (not exist_0) or \
            (not exist_1):
        print "'" + path_0 + "' : is not existed!"
        print "'" + path_1 + "' : is not existed!"
        print '***************'
        break

    OutputData(fd_val_train_list, path_0)
    OutputData(fd_val_train_list, path_1)

    print "Finish: " + val_folder_list[i]

# if __name__ == '__main__':
