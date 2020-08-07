# -*- coding: utf-8 -*-
from Basic.Switch import Switch
from Basic.LogHandler import *
from Basic.Define import *
from JackBasicStructLib.Basic.Paras import *
from Network.NLCANet.Model import NLCANet as nlca_v1
from Network.NLCANet_cat_val_spn.Model import NLCANet as nlca_v2
from Network.NLCANet_gn_spn.Model import NLCANet as nlca_v3
from Dataloader.Dataloader import *


class NetWorkInference(object):
    def __init__(self):
        pass

    def Inference(self, args, is_training=True):
        name = args.modelName
        for case in Switch(name):
            if case('NLCANet'):
                Info("Begin loading NLCA_Net Model")
                paras = self.__Args2Paras(args, is_training)
                model = nlca_v1(args, is_training)
                dataHandler = DataHandler(args)
                break
            if case('NLCANet_v2'):
                Info("Begin loading NLCA_Net_v2 Model")
                paras = self.__Args2Paras(args, is_training)
                model = nlca_v2(args, is_training)
                dataHandler = DataHandler(args)
                break
            if case('NLCANet_v3'):
                Info("Begin loading NLCA_Net_v3 Model")
                paras = self.__Args2Paras(args, is_training)
                model = nlca_v3(args, is_training)
                dataHandler = DataHandler(args)
                break
            if case():
                Error('NetWork Type Error!!!')

        return paras, model, dataHandler

    def __Args2Paras(self, args, is_training):
        paras = Paras(args.learningRate, args.batchSize,
                      args.gpu, args.imgNum,
                      args.valImgNum, args.maxEpochs,
                      args.log, args.modelDir,
                      MODEL_NAME, args.auto_save_num,
                      10, args.pretrain,
                      1, is_training)
        return paras
