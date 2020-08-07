#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=7 nohup python -u ./Source/main.py \
                       --gpu 1 --phase test \
					   --dataset Middlebury \
                       --modelDir ./PAModel_ROB_GN_Kitti/ \
                       --imgNum 15 \
                       --outputDir ./TestResult/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/trainlist_MiddEval3_H.txt \
                       --padedImgWidth 1536 \
                       --padedImgHeight 1024 \
					   --batchSize 1 \
                       --pretrain false > TestRun.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TestRun.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
