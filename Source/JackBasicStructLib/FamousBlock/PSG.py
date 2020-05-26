# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def ReadImg(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img


class PSG(object):

    """docstring for PSG"""

    def __init__(self, args):
        super(PSG, self).__init__()
        self.args = args
        pass

    def __SGM(self, imgL, imgR, disp_num):
        stereo = cv2.StereoSGBM_create(minDisparity=0,
                                       numDisparities=disp_num,
                                       blockSize=5,
                                       P1=0,
                                       P2=0,
                                       disp12MaxDiff=1,
                                       preFilterCap=255,
                                       uniquenessRatio=15,
                                       speckleWindowSize=0,
                                       speckleRange=2,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def Interface(self, imgL, imgR, disp_num):
        dsp = self.__SGM(imgL, imgR, disp_num)
        dsp = np.array(dsp)
        dsp = (dsp * float(256.0)).astype(np.uint16)
        cv2.imwrite('~/1.png', dsp)
        return dsp


if __name__ == '__main__':
    pathL = '/Users/rhc/000000_10_L.png'
    pathR = '/Users/rhc/000000_10_R.png'

    imgL = ReadImg(pathL)
    imgR = ReadImg(pathR)

    plt.imshow(imgL, 'gray')
    plt.show()

    plt.imshow(imgR, 'gray')
    plt.show()

    imgL = imgL[100:612, 100:356, :]
    imgR = imgR[100:612, 100:356, :]
    imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)

    plt.imshow(imgL, 'gray')
    plt.show()

    plt.imshow(imgR, 'gray')
    plt.show()

    net = PSG(None)
    dsp = net.Interface(imgL, imgR, 192)

    plt.imshow(dsp, 'gray')
    plt.show()
