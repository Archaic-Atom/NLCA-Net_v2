# -*- coding: utf-8 -*-
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.DataHandlerTemplate import DataHandlerTemplate
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.Dataloader.KittiFlyingDataloader import KittiFlyingDataloader as kfd
from JackBasicStructLib.ImgProc.ImgHandler import *
import time
import cv2
import numpy as np

TRAIN_ACC_FILE = 'train_acc.csv'                        # acc file's name
TRAIN_LOSS_FILE = 'train_loss.csv'                      # loss file's name
VAL_LOSS_FILE = 'val_loss.csv'                          # val file's name
VAL_ACC_FILE = 'val_acc.csv'                            # val file's name
TEST_ACC_FILE = 'test_acc.csv'                          # test file's name


class DataHandler(DataHandlerTemplate):
    """docstring for DataHandler"""

    def __init__(self, args):
        super(DataHandler, self).__init__()
        self.__args = args
        self.fd_train_acc, self.fd_train_loss, self.fd_val_acc,\
            self.fd_val_loss, self.fd_test_acc = self.__CreateResultFile(args)
        self.kfd = kfd()

    def GetTrainingData(self, paras, trainList, num):
        imgLs, imgRs, imgGrounds = self.kfd.GetBatchImage(self.__args, trainList, num)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetValData(self, paras, valList, num):
        imgLs, imgRs, imgGrounds = self.kfd.GetBatchImage(self.__args, valList, num, True)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetTestingData(self, paras, testList, num):
        imgLs, imgRs, top_pads, left_pads, names = self.kfd.GetBatchTestImage(
            self.__args, testList, num, True)
        input, _ = self.__CreateRes(imgLs, imgRs, None)
        supplement = self.__CreateSupplement(top_pads, left_pads, names)
        self.start_time = time.time()
        return input, supplement

    def ShowTrainingResult(self, epoch, loss, acc, duration):
        format_str = ('[TrainProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, cspn_disp_loss = %.6f, ' +
                      'refine_disp_loss = %.6f, coarse_acc = %.6f, ' +
                      'cspn_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, loss[0], loss[1], loss[2], loss[3],
                           acc[0], acc[1], acc[2], duration))
        OutputData(self.fd_train_acc, loss[0])
        OutputData(self.fd_train_loss, acc[1])

    def ShowValResult(self, epoch, loss, acc, duration):
        format_str = ('[ValProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, cspn_disp_loss = %.6f, ' +
                      'refine_disp_loss = %.6f, coarse_acc = %.6f, ' +
                      'cspn_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, loss[0], loss[1], loss[2], loss[3],
                           acc[0], acc[1], acc[2], duration))
        OutputData(self.fd_val_acc, loss[0])
        OutputData(self.fd_val_loss, acc[1])

    def ShowIntermediateResult(self, epoch, loss, acc):
        format_str = ('e: %d, l: %.3f, ' +
                      'l0: %.3f, l1: %.3f, l2: %.3f, ' +
                      'a0: %.3f, a1: %.3f, a2: %.3f')
        info_str = format_str % (epoch, loss[0], loss[1], loss[2], loss[3],
                                 acc[0], acc[1], acc[2])
        return info_str

    def SaveResult(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]
        ttimes = time.time() - self.start_time

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes = res[i, 2, j, :, :]

                top_pad = top_pads[i*args.batchSize+j]
                left_pad = left_pads[i*args.batchSize+j]

                if top_pad > 0 and left_pad > 0:
                    temRes = temRes[top_pad:, : -left_pad]
                elif top_pad > 0:
                    temRes = temRes[top_pad:, :]
                elif left_pad > 0:
                    temRes = temRes[:, :-left_pad]

                if args.dataset == "KITTI":
                    self.kfd.SaveTestData(args, temRes, args.gpu*args.batchSize *
                                          imgID + i*args.batchSize + j)
                elif args.dataset == "ETH3D":
                    name = names[i*args.batchSize+j]
                    path = args.resultImgDir + name + '.pfm'
                    WritePFM(path, temRes)

                    path = args.resultImgDir + name + '.txt'
                    with open(path, 'w') as f:
                        f.write("runtime "+str(ttimes))
                elif args.dataset == "Middlebury":
                    name = names[i*args.batchSize+j]
                    folder_name = args.resultImgDir + name + '/'
                    Mkdir(folder_name)
                    method_name = "disp0NLCA_NET_v2_RVC.pfm"
                    path = folder_name + method_name
                    WritePFM(path, temRes)

                    time_name = "timeNLCA_NET_v2_RVC.txt"
                    path = folder_name + time_name
                    with open(path, 'w') as f:
                        f.write(str(ttimes))

                    # Info('[TestProcess] Finish ' +
                    #     str(args.gpu * args.batchSize*imgID + i*args.batchSize + j) + ' image.')

    def __CreateRes(self, imgLs, imgRs, imgGrounds):
        input = []
        label = []
        input.append(imgLs)
        input.append(imgRs)
        label.append(imgGrounds)
        return input, label

    def __CreateSupplement(self, top_pads, left_pads, names):
        supplement = []
        supplement.append(top_pads)
        supplement.append(left_pads)
        supplement.append(names)
        return supplement

    def __CreateResultFile(self, args):
        # create the dir
        Info("Begin create the result folder")
        Mkdir(args.outputDir)
        Mkdir(args.resultImgDir)

        fd_train_acc = OpenLogFile(args.outputDir + TRAIN_LOSS_FILE, args.pretrain)
        fd_train_loss = OpenLogFile(args.outputDir + TRAIN_ACC_FILE, args.pretrain)
        fd_val_acc = OpenLogFile(args.outputDir + VAL_ACC_FILE, args.pretrain)
        fd_val_loss = OpenLogFile(args.outputDir + VAL_LOSS_FILE, args.pretrain)
        fd_test_acc = OpenLogFile(args.outputDir + TEST_ACC_FILE, args.pretrain)

        Info("Finish create the result folder")
        return fd_train_acc, fd_train_loss, fd_val_acc, fd_val_loss, fd_test_acc
