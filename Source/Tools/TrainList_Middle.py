# -*- coding: utf-8 -*-

import os
import glob


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


TrainListPath = './Dataset/trainlist_MiddEval3.txt'
DispLabelListPath = './Dataset/labellist_disp_MiddEval3.txt'

ValTrainListPath = './Dataset/testlist_MiddEval3.txt'
#ValDispLabelListPath = './Dataset/test_label_disp_list_ETH3D.txt'
#ValClsLabelListPath = './Dataset/test_label_cls_list_CityScape.txt'


RootPath = '/home1/Documents/Database/MiddEval/MiddEval3/'

train_folder = 'trainingQ/'
test_folder = 'testQ/'


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
