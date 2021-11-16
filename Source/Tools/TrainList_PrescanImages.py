# -*- coding: utf-8 -*-
import os
import glob


def OpenFile(path):
    if os.path.exists(path):
        os.remove(path)

    fd_test_list = open(path, 'a')

    return fd_test_list


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def GenList(left_imgs_folder_path, right_imgs_folder_path, raw_data_type, fd_test_list):
    total = 0
    left_files = glob.glob(left_imgs_folder_path + '*' + raw_data_type)
    for i in range(len(left_files)):
        name = os.path.basename(left_files[i])

        left_img_path = left_files[i]
        right_img_path = right_imgs_folder_path + name
        rawLeftPathisExists = os.path.exists(left_img_path)
        rawRightPathisExists = os.path.exists(right_img_path)

        if (not rawLeftPathisExists) or (not rawRightPathisExists):
            print("\"" + left_img_path + "\"" + "is not exist!!!")
            break

        OutputData(fd_test_list, left_img_path)
        OutputData(fd_test_list, right_img_path)

        total = total + 1

    return total


def main():
    fd_test_list = OpenFile('./Dataset/testlist_prescan_images_2.txt')
    total = GenList('/home4/datasets/jack/perscan_images_2/CameraSensor_1_Left/',
                    '/home4/datasets/jack/perscan_images_2/CameraSensor_1_Right/',
                    'jpg',
                    fd_test_list
                    )
    print(total)


if __name__ == '__main__':
    main()
