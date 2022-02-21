>This is the project of the StereoMatching Project. This project based on my framework (if you want to use it to build the Network, you can find it on my website: [fadeshine](http://www.fadeshine.com/). If you have any questions, you can send an e-mail to me. My e-mail: raoxi36@foxmail.com)

### Paper Information
@article{rao2022rethinking,

  title={Rethinking Training Strategy in Stereo Matching},
  
  author={Rao, Zhibo and Dai, Yuchao and Shen, Zhelun and He, Renjie},
  
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  
  year={2022},
  
  publisher={IEEE}
}

### Software Environment
1. OS Environment
os == linux 16.04
cudaToolKit == 10.1
cudnn == 7.3.6
2. Python Environment
python == 2.7.18
tensorflow == 1.12.0
numpy == 1.14.5
opencv == 3.4.0
PIL == 5.1.0

### Hardware Environment
- GPU: 1080TI * 4 or other memory at least 11G.(Batch size: 1)
if you not have four gpus, you could change the para of model. The Minimum hardware requirement:
- GPU: memory at least 5G. (Batch size: 1)

### Train the model by running:
1. Get the Training list or Testing list （You need rewrite the code by your path, and my related code can be found in Source/Tools）
```
$ ./GenPath.sh
```
Please check the path. The source code in Source/Tools.

2. Run the pre-training.sh (This is pre-training process. We will provide the pre-trained model at BaiduYun or Google Driver)
```
$ ./Pre-Train.sh
```
Please carefully check the path in related file.

3. Run the trainstart.sh (This is fine-tuing process.bg means background running. note that please check the img path should be found in related path, e.g. ./Dataset/trainlist_ETH3D.txt)
```
$ ./TrainKitti_2012_bg.sh
or
$ ./TrainKitti_2015_bg.sh
or
$ ./TrainKitti_ROB_bg.sh
```
Please carefully check the path in related file.

4. Run the teststart.sh ()
```
$ ./TestKitti2012.sh
or 
$ ./TestKitti2015.sh
or 
$ ./TestETH3D.sh
or 
$ ./TestMiddle_test.sh (for test data)
or
$ ./TestMiddle_train.sh (for train data)
or 
$ ./TestSceneFlow.sh
```

if you want to change the para of the model, you can change the *.sh file. Such as:
```
$ vi TestStart.sh
or 
$ vi TestETH3D.sh
```

### File Struct
```
.
├── Source # source code
│   ├── Basic
│   ├── Evaluation
│   └── ...
├── Dataset # Get it by ./GenPath.sh, you need build folder
│   ├── label_scene_flow.txt
│   ├── trainlist_scene_flow.txt
│   └── ...
├── Result # The data of Project. Auto Bulid
│   ├── output.log
│   ├── train_acc.csv
│   └── ...
├── ResultImg # The image of Result. Auto Bulid
│   ├── 000001_10.png
│   ├── 000002_10.png
│   └── ...
├── PAModel # The saved model. Auto Bulid
│   ├── checkpoint
│   └── ...
├── log # The graph of model. Auto Bulid
│   ├── events.out.tfevents.1541751559.ubuntu
│   └── ...
├── GetPath.sh
├── Pre-Train.sh
├── TestStart.sh
├── TrainStart.sh
├── LICENSE
├── requirements.txt
└── README.md
```

### Update log
#### 2020-08-09
1. Write the readMe;
2. Refactor the code (Dataloader and BuiildGraph);

#### 2020-07-30
1. Refactor the code;
2. Merge the code of SegCCNet;
3. Merge the code of PANet;


#### 2020-07-15
1. Refactor the code;
2. Add the network folder;
3. Add the shell file to start programs by step.

#### 2020-06-15
1. Add the D data agumentation;
2. Balance the data of datasets;
3. Add the variance and concat;

#### 2020-06-08
1. Add spn;
2. Add Refine;
3. Add the number of disparity;
4. Refactor the code in the future.

#### 2020-05-30
1. Add gn;
2. Add the 3D gn;
3. Change All bn to gn.

#### 2020-05-08
1. New project for ROB;
2. Add Middlebury;
3. ADD ETH3D;

___

#### 2019-10-23 (v1)
1. Finsih refactoring job;
2. Add some files and change the Source/JackBasicStructLib

#### 2019-10-19 (New fork)
1. New project from nlca-net and jacklib projects;
2. Tested the project and make it work;
3. Add some files
4. The target of this project is to build the quantization network for stereo matching tasks.

___

#### 2019-06-17
1. CHanged the file path;
2. Finish review the code of jacklib

#### 2019-01-05
1. Fixed some bugs in random crop process;
2. Update the ReadMe

#### 2018-12-15
1. Add the requirements.txt and LICENSE;
2. Update the 3D module
3. In the feature, We will update refine network.

#### 2018-12-08
1. Change the ReadMe.md;
2. Update the loghangdler.py;
3. Add the building network process in the log file;
4. Fixed some bugs in log file.

#### 2018-12-07
1. Fixed the long time in builduing network during the testing;
2. Add the LICENSE
3. Add the requirenments.txt

#### 2018-11-11
1. Modify the README file.

#### 2018-11-11
1. Write the README file;
2. Fixed some Bugs;
3. Change tensorflow to 1.9.0.

#### 2018-11-08
1. Add Test.py file;
2. Add Switch.py file;
3. Fixed some bugs.

#### 2018-11-05
1. Add the GenPath.sh file;
2. Add Path tool to get the training or Testing list on scence flow or KITTI
3. Add attention moudle;
4. Add GN module;

#### 2018-11-01
1. Finish the StereMatchingNext;
2. Add some file. e.g. Pre-Train.sh

#### 2018-10-30
1. Change the input file;
2. Build the Net Work

#### 2018-10-15
1. Add Multi-GPU, Test the program by Sensitivity Project;

#### 2018-08-25
1. Build the new project;
2. Add some basic network struct;
3. Add the __init__.py
4. Change the file folder.
