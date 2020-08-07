#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u ./Source/main.py \
                      --gpu 4 --phase train \
                      --dataset KITTI \
                      --modelDir ./PAModel_ROB/ \
                      --auto_save_num 20 \
                      --imgNum 814 \
                      --valImgNum 0 \
                      --maxEpochs 100 \
                      --learningRate 0.00001 \
                      --outputDir ./Result_ROB/ \
                      --trainListPath ./Dataset/ROB_training_list_v2.txt \
                      --trainLabelListPath ./Dataset/ROB_traning_labels_list_v2.txt \
                      --valListPath ./Dataset/val_trainlist_kitti_2015.txt \
                      --valLabelListPath ./Dataset/val_labellist_kitti_2015.txt \
                      --corpedImgWidth 512 \
                      --corpedImgHeight 256 \
                      --batchSize 1 \
                      --pretrain false
echo $"You can get the running log via the command line that tail -f TrainKitti.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
