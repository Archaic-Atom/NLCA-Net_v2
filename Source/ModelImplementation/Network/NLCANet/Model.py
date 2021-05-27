# -*- coding: utf-8 -*-
from Basic.Define import *
from BasicModule import *
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.ModelTemplate import ModelTemplate
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from JackBasicStructLib.Evaluation.AdamBound import *
import math


class NLCANet(ModelTemplate):
    def __init__(self, args, training=True):
        self.__args = args
        self.input_imgL_id = 0
        self.input_imgR_id = 1
        self.label_disp_id = 0
        self.output_coarse_img_id = 0
        self.output_cspn_img_id = 1
        self.output_refine_img_id = 2

        self.max_height = 2112
        self.max_width = 2496
        self.patch_height = 352

        self.down_sampling_times = 16

        if training == True:
            self.height = args.corpedImgHeight
            self.width = args.corpedImgWidth
        else:
            self.height = args.padedImgHeight
            self.width = args.padedImgWidth

    def GenInputInterface(self):
        input = []

        args = self.__args
        imgL = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        input.append(imgL)

        imgR = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        input.append(imgR)

        return input

    def GenLabelInterface(self):
        label = []
        args = self.__args

        imgGround = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width))
        label.append(imgGround)

        return label

    def Optimizer(self, lr):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        #opt = AdaBoundOptimizer(learning_rate=lr, final_lr=100 * lr)
        return [opt]

    def OptimizerVarList(self):
        var_list = (tf.trainable_variables()
                    + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        assert set(var_list) == set(tf.trainable_variables())
        # print set(var_list)
        # print set(tf.trainable_variables())
        # var_list = [var for var in var_list if var.name.startswith('NLCANet')]
        # print set(var_list)
        # var_list = [var_list]

        var_list = [None]
        return var_list

    def Accuary(self, output, label):
        acc = []

        coarse_acc = MatchingAcc(output[self.output_coarse_img_id], label[self.label_disp_id])
        cspn_acc = MatchingAcc(output[self.output_cspn_img_id], label[self.label_disp_id])
        refine_acc = MatchingAcc(output[self.output_refine_img_id], label[self.label_disp_id])
        acc.append(coarse_acc[1])
        acc.append(cspn_acc[1])
        acc.append(refine_acc[1])

        return acc

    def Loss(self, output, label):
        loss = []
        loss_0 = MAE_Loss_v2(output[self.output_coarse_img_id], label[self.label_disp_id])
        loss_1 = MAE_Loss_v2(output[self.output_cspn_img_id], label[self.label_disp_id])
        loss_2 = MAE_Loss_v2(output[self.output_refine_img_id], label[self.label_disp_id])

        loss_3 = MAE_Loss(output[self.output_coarse_img_id], label[self.label_disp_id])
        loss_4 = MAE_Loss(output[self.output_cspn_img_id], label[self.label_disp_id])
        loss_5 = MAE_Loss(output[self.output_refine_img_id], label[self.label_disp_id])
        total_loss = loss_2 + loss_1 + loss_0
        loss.append(total_loss)
        loss.append(loss_3)
        loss.append(loss_4)
        loss.append(loss_5)
        return loss

    # This is the Inference, and you must have it!
    def Inference(self, input, training=True):
        imgL, imgR = self.__GetVar(input)
        coarse_map, output_cspn, refine_map = self.__NetWork(
            imgL, imgR, self.height, self.width, training)
        output = self.__GenRes(coarse_map, output_cspn, refine_map)
        return output

    def __NetWork(self, imgL, imgR, height, width, training=True):
        with tf.variable_scope("NLCANet"):
            Info('├── Begin Build ExtractUnaryFeature')
            with tf.variable_scope("ExtractUnaryFeature") as scope:
                imgL_feature = ExtractUnaryFeatureModule().Inference(imgL, training=training)
                scope.reuse_variables()
                imgR_feature = ExtractUnaryFeatureModule().Inference(imgR, training=training)
            Info('│   └── After ExtractUnaryFeature:' + str(imgL_feature.get_shape()))

            _, height, width, feature_num = imgL_feature.get_shape().as_list()

            assert ((height % self.down_sampling_times == 0)
                    and width % self.down_sampling_times == 0)

            patch_num = height // self.patch_height

            for i in range(patch_num):
                start_h = i * self.patch_height
                slice_features_l = tf.slice(imgL_feature, [0, start_h, 0, 0],
                                            [-1, self.patch_height, width, feature_num])
                slice_features_r = tf.slice(imgR_feature, [0, start_h, 0, 0],
                                            [-1, self.patch_height, width, feature_num])

                Info('├── Begin Build Cost Volume')
                tmp_cost_vol = BuildCostVolumeModule().Inference(slice_features_l,
                                                                 slice_features_r,
                                                                 IMG_DISPARITY,
                                                                 training=training)
                Info('│   └── After Cost Volume:' + str(tmp_cost_vol.get_shape()))

                Info('├── Begin Build 3DMatching')
                tmp_coarse_map, tmp_mask = MatchingModule().Inference(tmp_cost_vol,
                                                                      reliability=0.65,
                                                                      training=training)
                Info('│   └── After 3DMatching:' + str(tmp_coarse_map.get_shape()))

                if i == 0:
                    coarse_map = tmp_coarse_map
                    mask = tmp_mask
                else:
                    coarse_map = tf.concat([coarse_map, tmp_coarse_map], axis=1)
                    mask = tf.concat([mask, tmp_mask], axis=1)

            Info('│   └── After final 3DMatching:' + str(coarse_map.get_shape()))

            Info('├── Begin Build Guidance')
            cspn_guidance = GetGuidanceModule().Inference(imgL, training=training)
            Info('│   └── After Guidance:' + str(cspn_guidance.get_shape()))

            Info('├── Begin Build DispMapRefine')
            output_cspn = DispRefinementModule().Inference(coarse_map, mask,
                                                           cspn_guidance, training=training)
            Info('│   └── After DispMapRefine:' + str(output_cspn.get_shape()))

            Info('└── Begin Build Fusion')
            refine_map = FusionModule().Inference(coarse_map, output_cspn, imgL,
                                                  imgL_feature, training=training)
            Info('    └── After Fusion:' + str(refine_map.get_shape()))

            #mask = tf.squeeze(mask, axis=3)
            # refine_map = mask * coarse_map # confident map
        return coarse_map, output_cspn, refine_map

    def __GetVar(self, input):
        return input[self.input_imgL_id], input[self.input_imgR_id]

    def __GenRes(self, coarse_map, output_cspn, refine_map):
        res = []
        res.append(coarse_map)
        res.append(output_cspn)
        res.append(refine_map)
        return res
