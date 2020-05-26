# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *
from BasicBlock import *
from JackBasicStructLib.FamousBlock.CSPN import *


def ExtractUnaryFeatureModule(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureModule"):
        x = ExtractUnaryFeatureBlock1(x, training=training)
        output_raw = ExtractUnaryFeatureBlock2(x, training=training)
        output_skip_1 = ExtractUnaryFeatureBlock3(output_raw, training=training)
        output_skip_2 = ExtractUnaryFeatureBlock4(output_skip_1, training=training)
        x = ASPPBlock(output_skip_2, 32, "ASPP", training=training)
        x = tf.concat([output_raw, output_skip_1, output_skip_2, x], axis=3)
        x = ExtractUnaryFeatureBlock5(x, training=training)
    return x


def BuildCostVolumeModule(imgL, imgR, disp_num, training=True):
    with tf.variable_scope("BuildCostVolume") as scope:
        cost_vol = BuildCostVolumeBlock(imgL, imgR, disp_num)
    return cost_vol


def MatchingModule(x, reliability=0.65, training=True):
    with tf.variable_scope("MatchingModule"):
        x = FeatureMatchingBlock(x, training=training)
        x = RecoverSizeBlock(x, training=training)
        x, mask = SoftArgMinBlock(x, reliability=0.65)
    return x, mask


def GetGuidanceModule(x, training=True):
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


def AttentionModule(x, dsp_num, training=True):
    with tf.variable_scope("AttentionModule"):
        x = ExtractAttentionFeatureBlock(x, training=training)
        x = AttentionBlock(x, training=training)
        seg_attention = SegAttentionBlock(x, training=training)
        dsp_attention = DspAttentionBlock(x, dsp_num, training=training)

    return seg_attention, dsp_attention


def DispRefinementModule(x, sparse_depth, guidance, training=True):
    with tf.variable_scope("RefinementModule"):
        x = tf.expand_dims(x, axis=3)
        sparse_depth = tf.expand_dims(sparse_depth, axis=3)
        sparse_depth = sparse_depth * x
        x = CSPN(x, sparse_depth, guidance, kernel_size=3, num_layers=8)
        x = tf.squeeze(x, axis=3)
    return x


def FusionModule(x, output_cspn, imgL, seg, training=True):
    with tf.variable_scope("FusionModule"):
        shortcut = output_cspn
        x = tf.expand_dims(x, axis=3)
        output_cspn = tf.expand_dims(output_cspn, axis=3)
        x = tf.concat([x, output_cspn], axis=3)
        x = Conv2DLayer(x, 3, 1, 1, "Conv_2", biased=False,
                        relu=False, bn=False, training=training)
        x = tf.squeeze(x, axis=3)
        x = FeatureConcat(x, imgL, seg, training=training)
        x = ResidualLearning(x, training=training)
        x = shortcut + x
        #x = 0.5 * x + 0.5 * output_cspn
        #x = tf.squeeze(x, axis=3)
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
