# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
# Test the code of acc
import tensorflow as tf
import re
import cv2
import sys
import os
import linecache

DEPTH_DIVIDING = 256.0
DSP_NUM = 256


def ReadPFM(file):
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


def PaddingImg(img, h, w):

    # print img.shape

    top_pad = h - img.shape[0]
    left_pad = w - img.shape[1]

    # print top_pad
    # print left_pad

    # pading
    img = np.lib.pad(img, ((top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
    return img


def ReadRandomGroundTrue(path, h, w):
    # kitti groundtrue
    img = Image.open(path)
    imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
    # pading size
    imgGround = PaddingImg(imgGround, h, w)
    return imgGround


def ReadRandomPfmGroundTrue(path, h, w):
    # flying thing groundtrue
    imgGround, _ = ReadPFM(path)
    imgGround = PaddingImg(imgGround, h, w)
    return imgGround


def ReadDSP(path, h, w):
    file_type = os.path.splitext(path)[-1]
    if file_type == ".png":
        imgGround = ReadRandomGroundTrue(path, h, w)
    else:
        imgGround = ReadRandomPfmGroundTrue(path, h, w)

    return imgGround


def GetPath(filename, num):
    path = linecache.getline(filename, num)
    path = path.rstrip("\n")
    return path


def Count(img):
    img = tf.cast(img, dtype=tf.int32)
    res = []
    for i in range(DSP_NUM):
        tem_standard = tf.ones_like(img) * (i+2)
        tem_standard = tf.equal(tem_standard, img)
        tem_standard = tf.cast(tem_standard, dtype=tf.int32)
        tem_standard = tf.reduce_sum(tem_standard)
        res.append(tem_standard)
    res = tf.stack(res, axis=0)
    return res


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def OpenLogFile(path, is_continue=True):
    if is_continue == False:
        if os.path.exists(path):
            os.remove(path)

    file = open(path, 'a')

    return file


if __name__ == "__main__":
    h = 1024
    w = 1024
    path = '/home1/jack/Documents/Programs/QSMNet_ROB/Dataset/labellist_disp_ETH3D.txt'
    a = tf.placeholder(tf.float32, shape=(h, w))
    res = Count(a)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    img_num = 27
    fd_res = OpenLogFile('4.txt', False)

    for i in range(img_num):
        img_path = GetPath(path, i+1)
        # print img_path
        img = ReadDSP(img_path, h, w)
        # print img.shape
        x = sess.run(res, feed_dict={a: img})
        # print x
        #
        if i == 0:
            data_res = x
        else:
            data_res = data_res + x
        print i

    # print data_res
    for i in range(data_res.shape[0]):
        OutputData(fd_res, data_res[i])

    print "Finish!"
