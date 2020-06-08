# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import sys
from PIL import Image
import cv2


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()

    return data, scale


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


TrainListPath = './Dataset/trainlist_MiddEval3_H.txt'
DispLabelListPath = './Dataset/labellist_disp_MiddEval3_H.txt'

ValTrainListPath = './Dataset/testlist_MiddEval3_H.txt'
#ValDispLabelListPath = './Dataset/test_label_disp_list_ETH3D.txt'
#ValClsLabelListPath = './Dataset/test_label_cls_list_CityScape.txt'


RootPath = '/home1/Documents/Database/MiddEval/MiddEval3/'

train_folder = 'trainingH/'
test_folder = 'testH/'


#cls_folder_list = ['leftImg8bit/', 'rightImg8bit/', 'gtCoarse/', 'disparity/']


train_folder_list = ['Adirondack', 'Jadeplant', 'MotorcycleE', 'PianoL',
                     'Playroom', 'PlaytableP', 'Shelves', 'Vintage',
                     'ArtL', 'Motorcycle', 'Piano', 'Pipes',
                     'Playtable', 'Recycle', 'Teddy']

val_folder_list = ['Australia', 'Bicycle2', 'Classroom2E', 'Crusade',
                   'Djembe', 'Hoops', 'Newkuba', 'Staircase',
                   'AustraliaP', 'Classroom2', 'Computer', 'CrusadeP',
                   'DjembeL', 'Livingroom', 'Plants']


if os.path.exists(TrainListPath):
    os.remove(TrainListPath)

if os.path.exists(DispLabelListPath):
    os.remove(DispLabelListPath)

if os.path.exists(ValTrainListPath):
    os.remove(ValTrainListPath)

fd_train_list = open(TrainListPath, 'a')
fd_disp_label_list = open(DispLabelListPath, 'a')
fd_val_train_list = open(ValTrainListPath, 'a')


for i in range(len(train_folder_list)):
    path = RootPath + train_folder + train_folder_list[i]

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

    OutputData(fd_train_list, path_0)
    OutputData(fd_train_list, path_1)
    OutputData(fd_disp_label_list, path_2)

    disp, _ = readPFM(path_2)
    disp[np.isinf(disp)] = 0

    tem_disp = disp > 256
    print np.max(disp)
    tem_disp = tem_disp.astype(np.int32)
    tem_disp_1 = np.sum(tem_disp)
    # print tem_disp_1

    tem_disp = disp > 0
    tem_disp = tem_disp.astype(np.int32)
    tem_disp = np.sum(tem_disp)
    # print tem_disp

    print tem_disp_1 / float(tem_disp)

    print "Finish: " + train_folder_list[i]


for i in range(len(val_folder_list)):
    path = RootPath + test_folder + val_folder_list[i]

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
