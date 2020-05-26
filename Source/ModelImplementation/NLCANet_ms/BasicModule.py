# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *
from BasicBlock import *


def ExtractUnaryFeatureModule(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureModule"):
        x = ExtractUnaryFeatureBlock1(x, training=training)
        output_raw = ExtractUnaryFeatureBlock2(x, training=training)
        output_skip_1 = ExtractUnaryFeatureBlock3(output_raw, training=training)
        output_skip_2 = ExtractUnaryFeatureBlock4(output_skip_1, training=training)
        x = ASPPBlock(output_skip_2, 32, "ASPP", training=training)
        x = tf.concat([output_raw, output_skip_1, output_skip_2, x], axis=3)
        x_0 = ExtractUnaryFeatureBlock5(x, training=training)
        x_1 = ExtractUnaryFeatureBlock6(x_0, training=training)
        x_2 = ExtractUnaryFeatureBlock7(x_1, training=training)
        x_3 = ExtractUnaryFeatureBlock8(x_2, training=training)
        x_4 = ExtractUnaryFeatureBlock9(x_3, training=training)

        multi_x = []
        multi_x.append(x_0)
        multi_x.append(x_1)
        multi_x.append(x_2)
        multi_x.append(x_3)
        multi_x.append(x_4)
    return multi_x


def BuildCostVolumeModule(imgL, imgR, disp_num, training=True):
    with tf.variable_scope("BuildCostVolume") as scope:
        multi_cost_vol = []
        cost_vol_0 = BuildCostVolumeBlock(imgL[0], imgR[0], disp_num, "cost_0")
        cost_vol_1 = BuildCostVolumeBlock(imgL[1], imgR[1], disp_num / 2, "cost_1")
        cost_vol_2 = BuildCostVolumeBlock(imgL[2], imgR[2], disp_num / 4, "cost_2")
        cost_vol_3 = BuildCostVolumeBlock(imgL[3], imgR[3], disp_num / 8, "cost_3")
        cost_vol_4 = BuildCostVolumeBlock(imgL[4], imgR[4], disp_num / 16, "cost_4")
        multi_cost_vol.append(cost_vol_0)
        multi_cost_vol.append(cost_vol_1)
        multi_cost_vol.append(cost_vol_2)
        multi_cost_vol.append(cost_vol_3)
        multi_cost_vol.append(cost_vol_4)
    return multi_cost_vol


def MatchingModule(multi_cost_vol, training=True):
    with tf.variable_scope("MatchingModule"):
        x = FeatureMatchingBlock(multi_cost_vol, training=training)
        x = RecoverSizeBlock(x, training=training)
        x = SoftArgMinBlock(x)
    return x


def AttentionModule(x, dsp_num, training=True):
    with tf.variable_scope("AttentionModule"):
        x = ExtractAttentionFeatureBlock(x, training=training)
        x = AttentionBlock(x, training=training)
        seg_attention = SegAttentionBlock(x, training=training)
        dsp_attention = DspAttentionBlock(x, dsp_num, training=training)

    return seg_attention, dsp_attention


def DispRefinementModule(x, imgL, seg, training=True):
    with tf.variable_scope("RefinementModule"):
        shortcut = x
        x = FeatureConcat(x, imgL, seg[0], training=training)
        x = ResidualLearning(x, training=training)
        x = shortcut + x
    return x


def SegModule(x, seg_attention, cls_num, training=True):
    with tf.variable_scope("SegModule"):
        shortcut = ExtractSegFeatureBlock(x, training=training)
        x = SPPBlock(shortcut, 32, "SPP", training=training)
        x = tf.concat([x, shortcut], axis=3)
        x = FeatureFusionBlock(x, training=training)
        x = SegAttentionAddBlock(x, seg_attention, training=training)
        x = RecoveryCLSSizeBlock(x, cls_num, training=training)
    return x


def ClsRefinementModule(x, imgL, disp, training=True):
    with tf.variable_scope("ClsRefinementModule"):
        shortcut = x
        cls_num = x.get_shape()[3]
        x = ClsFeatureConcat(x, imgL, disp, training=training)
        x = ClsResidualLearning(x, cls_num, training=training)
        x = shortcut + x

    return x
