# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *
from BasicBlock import *
from JackBasicStructLib.FamousBlock.CSPN import *


class ExtractUnaryFeatureModule(object):
    """docstring for ClassName"""

    def __init__(self, arg=None):
        super(ExtractUnaryFeatureModule, self).__init__()
        self.arg = arg

    def Inference(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureModule"):
            x = self.__ExtractUnaryFeatureBlock1(x, training=training)
            output_raw = self.__ExtractUnaryFeatureBlock2(x, training=training)
            output_skip_1 = self.__ExtractUnaryFeatureBlock3(output_raw, training=training)
            output_skip_2 = self.__ExtractUnaryFeatureBlock4(output_skip_1, training=training)
            x = ASPPBlock(output_skip_2, 32, "ASPP", training=training)
            x = tf.concat([output_raw, output_skip_1, output_skip_2, x], axis=3)
            x = self.__ExtractUnaryFeatureBlock5(x, training=training)
        return x

    def __ExtractUnaryFeatureBlock1(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock1"):
            x = Conv2DLayer(x, 3, 2, 32, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, 32, "Conv_2", training=training)
            x = Conv2DLayer(x, 3, 1, 32, "Conv_3", training=training)

            res_block_num = 3
            for i in range(res_block_num):
                x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

        return x

    def __ExtractUnaryFeatureBlock2(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock2"):
            shortcut = Conv2DLayer(x, 3, 2, 64, "Conv_1", training=training)
            x = Conv2DLayer(x, 1, 2, 64, "Conv_w", training=training)
            x = tf.add(x, shortcut)

            res_block_num = 16
            for i in range(res_block_num):
                x = Bottleneck2DBlock(x, "BottleNeck_" + str(i), training=training)

        return x

    def __ExtractUnaryFeatureBlock3(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock3"):
            x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
            x = ResAtrousBlock(x, 3, 2, "Atrous_1", training=training)

        return x

    def __ExtractUnaryFeatureBlock4(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock4"):
            x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
            x = ResAtrousBlock(x, 3, 4, "Atrous_1", training=training)
        return x

    def __ExtractUnaryFeatureBlock5(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock5"):
            x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, 64, "Conv_2", biased=True,
                            bn=False, relu=False, training=training)
        return x


class BuildCostVolumeModule(object):
    """docstring for BuildCostVolumeModule"""

    def __init__(self, arg=None):
        super(BuildCostVolumeModule, self).__init__()
        self.arg = arg

    def Inference(self, imgL, imgR, disp_num, training=True):
        with tf.variable_scope("BuildCostVolume") as scope:
            cost_vol = self.__BuildCostVolumeBlock(imgL, imgR, disp_num)
        return cost_vol

    def __BuildCostVolumeBlock(self, imgL, imgR, disp_num):
        with tf.variable_scope("BuildCostVolumeBlock"):
            batchsize, height, width, feature_num = imgL.get_shape().as_list()
            cost_vol = []
            for d in xrange(1, int(disp_num/4) + 1):
                paddings = [[0, 0], [0, 0], [d, 0], [0, 0]]
                slice_featuresR = tf.slice(imgR, [0, 0, 0, 0],
                                           [-1, height, width - d, feature_num])
                slice_featuresR = tf.pad(slice_featuresR, paddings, "CONSTANT")
                ave_feature = tf.add(imgL, slice_featuresR) / 2
                ave_feature2 = tf.add(tf.square(imgL), tf.square(slice_featuresR)) / 2
                cost = ave_feature2 - tf.square(ave_feature)
                # cost = tf.concat([imgL, slice_featuresR], axis=3)
                cost_vol.append(cost)

            cost_vol = tf.stack(cost_vol, axis=1)
        return cost_vol


class MatchingModule(object):
    """docstring for MatchingModule"""

    def __init__(self, arg=None):
        super(MatchingModule, self).__init__()
        self.arg = arg

    def Inference(self, x, reliability=0.65, training=True):
        with tf.variable_scope("MatchingModule"):
            x = self.__FeatureMatchingBlock(x, training=training)
            x = self.__RecoverSizeBlock(x, training=training)
            x, mask = self.__SoftArgMinBlock(x, reliability=0.65)
        return x, mask

    def __FeatureMatchingBlock(self, x, training=True):
        with tf.variable_scope("FeatureMatchingBlock"):
            x = self.__MatchingBlock(x, training=training)
            shortcut = x
            x, level_list = self.__EncoderBlock(x, training=training)
            x = self.__NonLocalGroupBlock(x, training=training)
            x = self.__DecoderBlock(x, level_list, training=training)
            x = tf.add(shortcut, x)
        return x

    def __MatchingBlock(self, x, training=True):
        with tf.variable_scope("MatchingBlock"):
            x = Conv3DLayer(x, 3, 1, 64, "Conv_1", training=training)
            x = Conv3DLayer(x, 3, 1, 32, "Conv_2", training=training)
            x = Res3DBlock(x, 3, "Res_1", training=training)
        return x

    def __EncoderBlock(self, x, training=True):
        with tf.variable_scope("EncoderBlock"):
            level_list = []
            # 1/8
            x = DownSamplingBlock(x, 3, 32, "Level_1", training=training)
            level_list.append(x)

            # 1/16
            x = DownSamplingBlock(x, 3, 64, "Level_2", training=training)
            level_list.append(x)

            # 1/32
            x = DownSamplingBlock(x, 3, 96, "Level_3", training=training)
            level_list.append(x)

            # 1/64
            x = DownSamplingBlock(x, 3, 128, "Level_4", training=training)
        return x, level_list

    def __NonLocalGroupBlock(self, x, training=True):
        with tf.variable_scope("NonLocalGroupBlock"):
            x = SpaceTimeNonlocalBlock(x, "NonLocal_0", training=training)
            x = SpaceTimeNonlocalBlock(x, "NonLocal_1", training=training)
            x = SpaceTimeNonlocalBlock(x, "NonLocal_2", training=training)
            #x = SpaceTimeNonlocalBlock(x, "NonLocal_3", training=training)
            #x = SpaceTimeNonlocalBlock(x, "NonLocal_4", training=training)

        return x

    def __DecoderBlock(self, x, level_list, training=True):
        with tf.variable_scope("DecoderBlock"):
            # 1/32
            x = UpSamplingBlock(x, 3, 96, "Level_3", training=training)
            x = tf.add(x, level_list[2])

            # 1/16
            x = UpSamplingBlock(x, 3, 64, "Level_2", training=training)
            x = tf.add(x, level_list[1])

            # 1/8
            x = UpSamplingBlock(x, 3, 32, "Level_1", training=training)
            x = tf.add(x, level_list[0])

            # 1/4
            x = UpSamplingBlock(x, 3, 32, "Level_0", training=training)

        return x

    def __RecoverSizeBlock(self, x, training=True):
        with tf.variable_scope("RecoverSizeBlock"):
            # 1/2
            x = DeConv3DLayer(x, 3, 2, 8, "DeConv_1", training=training)
            x = DeConv3DLayer(x, 3, 2, 1, "DeConv_2", biased=True,
                              relu=False, bn=False, training=training,)
            x = Conv3DLayer(x, 3, 1, 1, "Conv_1", biased=True,
                            relu=False, bn=False, training=training)
        return x

    def __GetWeightBlock(self, batchsize, disp_num, height, width):
        disp = tf.range(0, disp_num, 1.0)
        disp = tf.cast(disp, tf.float32)
        w = tf.reshape(disp, [1, disp_num, 1, 1])
        w = tf.tile(w, [batchsize, 1, height, width])
        return w

    def __SoftArgMinBlock(self, x, reliability=0.65):
        with tf.variable_scope("SoftArgMin"):
            x = tf.squeeze(x, axis=4)
            batchsize, disp_num, height, width = x.get_shape().as_list()
            w = self.__GetWeightBlock(batchsize, disp_num, height, width)
            prob = tf.nn.softmax(x, axis=1)
            x = tf.multiply(prob, w)
            x = tf.reduce_sum(x, axis=1)
            mask = prob > reliability
            mask = tf.cast(mask, tf.float32)
            mask = tf.reduce_sum(mask, axis=1)
        return x, mask


class GetGuidanceModule(object):
    """docstring for GetGuidanceModule"""

    def __init__(self, arg=None):
        super(GetGuidanceModule, self).__init__()
        self.arg = arg

    def Inference(self, x, training=True):
        with tf.variable_scope("GetGuidanceModule"):
            x = Conv2DLayer(x, 3, 1, 32, "Conv_0", training=training)

            level_1 = Conv2DLayer(x, 3, 2, 64, "Conv_1", training=training)
            level_1 = Bottleneck2DBlock(level_1, "BottleNeck_1", training=training)

            level_2 = Conv2DLayer(level_1, 3, 2, 128, "Conv_2", training=training)
            level_2 = Bottleneck2DBlock(level_2, "BottleNeck_2", training=training)

            level_3 = Conv2DLayer(level_2, 3, 2, 192, "Conv_3", training=training)
            level_3 = Bottleneck2DBlock(level_3, "BottleNeck_3", training=training)

            delevel_3 = DeConv2DLayer(level_3, 3, 2, 128, "DeConv_1", training=training)
            delevel_3 = tf.add(level_2, delevel_3)

            delevel_2 = DeConv2DLayer(delevel_3, 3, 2, 64, "DeConv_2", training=training)
            delevel_2 = tf.add(level_1, delevel_2)

            delevel_1 = DeConv2DLayer(delevel_2, 3, 2, 32, "DeConv_3", training=training)
            x = tf.add(x, delevel_1)

            res_block_num = 3
            for i in range(res_block_num):
                x = GCBlock(x, "GCBlock_" + str(i), training=training)
                x = Bottleneck2DBlock(x, "GC_BottleNeck_" + str(i), training=training)

            x = Conv2DLayer(x, 3, 1, 8, "Conv_4", biased=True,
                            relu=False, bn=False, training=training)
        return x


class DispRefinementModule(object):
    """docstring for ClassName"""

    def __init__(self, arg=None):
        super(DispRefinementModule, self).__init__()
        self.arg = arg

    def Inference(self, x, sparse_depth, guidance, training=True):
        with tf.variable_scope("RefinementModule"):
            x = tf.expand_dims(x, axis=3)
            sparse_depth = tf.expand_dims(sparse_depth, axis=3)
            sparse_depth = sparse_depth * x
            x = CSPN(x, sparse_depth, guidance, kernel_size=3, num_layers=8)
            x = tf.squeeze(x, axis=3)
        return x


class FusionModule(object):
    """docstring for ClassName"""

    def __init__(self, arg=None):
        super(FusionModule, self).__init__()
        self.arg = arg

    def Inference(self, x, output_cspn, imgL, seg, training=True):
        with tf.variable_scope("FusionModule"):
            shortcut = output_cspn
            x = tf.expand_dims(x, axis=3)
            output_cspn = tf.expand_dims(output_cspn, axis=3)
            x = tf.concat([x, output_cspn], axis=3)
            x = Conv2DLayer(x, 3, 1, 1, "Conv_2", biased=False,
                            relu=False, bn=False, training=training)
            x = tf.squeeze(x, axis=3)
            x = FeatureConcat(x, imgL, seg, training=training)
            x = self.__ResidualLearning(x, training=training)
            x = shortcut + x
            # x = 0.5 * x + 0.5 * output_cspn
            # x = tf.squeeze(x, axis=3)
        return x

    def __ResidualLearning(self, x, training=True):
        with tf.variable_scope("ResidualLearning"):
            x = Conv2DLayer(x, 3, 1, 32, "Conv_1", training=training)

            res_block_num = 3
            for i in range(res_block_num):
                x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

            x = Conv2DLayer(x, 3, 1, 1, "Conv_2", biased=True,
                            relu=False, bn=False, training=training)
            x = tf.squeeze(x, axis=3)
        return x
