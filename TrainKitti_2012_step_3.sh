#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u ./Source/main.py \
                      --gpu 4 --phase train \
                      --dataset KITTI \
                      --modelDir ./PAModel_ROB/ \
                      --auto_save_num 20 \
                      --imgNum 194 \
                      --valImgNum 0 \
                      --maxEpochs 100 \
                      --learningRate 0.00001 \
                      --outputDir ./Result_KITTI2012/ \
                      --trainListPath ./Dataset/trainlist_kitti_2012.txt \
                      --trainLabelListPath ./Dataset/labellist_kitti_2012.txt \
                      --corpedImgWidth 512 \
                      --corpedImgHeight 256 \
                      --batchSize 1 \
                      --pretrain false
echo $"You can get the running log via the command line that tail -f TrainKitti.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
