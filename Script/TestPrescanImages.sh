#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=7 nohup python -u ./Source/main.py \
                       --gpu 1 --phase test \
                       --modelDir ./PAModel_ROB_cat_val_gn_mnl_KITTI_ETH3D_Middle_1.99_360/ \
                       --imgNum 2096 \
                       --padedImgWidth 960 \
                       --padedImgHeight 768 \
                       --outputDir ./TestResult/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/testlist_prescan_images.txt \
                       --batchSize 1 \
                       --pretrain false > TestRun.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TestRun.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
