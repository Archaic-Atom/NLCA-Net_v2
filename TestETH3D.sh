#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=2 nohup python -u ./Source/main.py \
                       --gpu 1 --phase test \
					   --dataset ETH3D \
                       --modelDir ./PAModel_ROB/ \
                       --imgNum 47 \
                       --outputDir ./TestResult/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/testlist_ETH3D.txt \
                       --padedImgWidth 960 \
                       --padedImgHeight 576 \
					   --batchSize 1 \
                       --pretrain false > TestRun.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TestRun.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
