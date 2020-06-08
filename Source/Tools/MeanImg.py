# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image


# output file setting
DEPTH_DIVIDING = 256.0


def DepthToImgArray(img):
    img = np.array(img)
    img = (img * float(DEPTH_DIVIDING)).astype(np.uint16)
    return img


def SavePngImg(path, img):
    cv2.imwrite(path, img)


def ReadRandomGroundTrue(path):
    # kitti groundtrue
    img = Image.open(path)
    imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
    return imgGround


if __name__ == "__main__":
    path_1 = '/Users/rhc/disp_0_2012_123/'
    path_2 = '/Users/rhc/disp_0_2012_125/'
    path_3 = '/Users/rhc/disp_0/'

    img_format = "%06d_10.png"
    img_num = 195

    for i in range(img_num):
        path = path_1 + img_format % i
        img_1 = ReadRandomGroundTrue(path)
        path = path_2 + img_format % i
        img_2 = ReadRandomGroundTrue(path)
        #img_3 = 0.7 * img_1 + 0.3 * img_2
        #img_3 = DepthToImgArray(img_3)
        print img_1.shape
        img_3 = img_1[:, 300:]
        print img_1.shape
        img_4 = img_2[:, :300]
        img_4 = 0.9*img_4 + 0.1*img_1[:, :300]
        print img_2.shape
        img_3 = np.concatenate((img_4, img_3), axis=1)
        path = path_3 + img_format % i
        img_3 = DepthToImgArray(img_3)
        SavePngImg(path, img_3)
