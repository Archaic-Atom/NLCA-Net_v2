# -*- coding: utf-8 -*-
from JackBasicStructLib.ImgProc.ImgHandler import *
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.ImgProc.DataAugmentation import *


# output file setting
DEPTH_DIVIDING = 256.0


class KittiFlyingDataloader(object):
    def __init__(self):
        super(KittiFlyingDataloader, self).__init__()
        self.imgL = None
        self.imgR = None
        pass

    def SaveKITTITestData(self, args, img, num):
        path = self.__GenerateOutImgPath(args.resultImgDir, args.saveFormat, args.imgType, num)
        img = self.__DepthToImgArray(img)
        self.__SavePngImg(path, img)

    def SaveETH3DTestData(self, args, img, name, ttimes):
        path = args.resultImgDir + name + '.pfm'
        WritePFM(path, img)
        path = args.resultImgDir + name + '.txt'
        with open(path, 'w') as f:
            f.write("runtime "+str(ttimes))
            f.close()

    def SaveMiddleburyTestData(self, args, img, name, ttimes):
        folder_name = args.resultImgDir + name + '/'
        Mkdir(folder_name)
        method_name = "disp0NLCA_NET_v2_RVC.pfm"
        path = folder_name + method_name
        WritePFM(path, img)

        time_name = "timeNLCA_NET_v2_RVC.txt"
        path = folder_name + time_name
        with open(path, 'w') as f:
            f.write(str(ttimes))
            f.close()

    def CropTestImg(self, img, top_pad, left_pad):
        if top_pad > 0 and left_pad > 0:
            img = img[top_pad:, : -left_pad]
        elif top_pad > 0:
            img = img[top_pad:, :]
        elif left_pad > 0:
            img = img[:, :-left_pad]
        return img

    def GetBatchImage(self, args, randomlist, num, isVal=False):
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]

            if isVal == False:
                imgL, imgR, imgGround = self.__RandomCropRawImage(args, idNum)       # get img
            else:
                imgL, imgR, imgGround = self.__ValRandomCropRawImage(args, idNum)       # get img

            if i == 0:
                imgLs = imgL
                imgRs = imgR
                imgGrounds = imgGround
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)
                imgGrounds = np.concatenate((imgGrounds, imgGround), axis=0)

        return imgLs, imgRs, imgGrounds

    def GetBatchTestImage(self, args, randomlist, num, isVal=False):
        top_pads = []
        left_pads = []
        names = []
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]
            imgL, imgR, top_pad, left_pad, name = self.__GetPadingTestData(
                args, idNum)       # get img

            top_pads.append(top_pad)
            left_pads.append(left_pad)
            names.append(name)
            if i == 0:
                imgLs = imgL
                imgRs = imgR
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)

        return imgLs, imgRs, top_pads, left_pads, names

    def __GenerateOutImgPath(self, dirPath, filenameFormat, imgType, num):
        path = dirPath + filenameFormat % num + imgType
        return path

    def __DepthToImgArray(self, img):
        img = np.array(img)
        img = (img * float(DEPTH_DIVIDING)).astype(np.uint16)
        return img

        # save the png file
    def __SavePngImg(self, path, img):
        cv2.imwrite(path, img)

    def __ReadRandomPfmGroundTrue(self, path, x, y, w, h):
        # flying thing groundtrue
        imgGround, _ = ReadPFM(path)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        # imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadRandomGroundTrue(self, path, x, y, w, h):
        # kitti groundtrue
        img = Image.open(path)
        imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        # imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadData(self, args, pathL, pathR, pathGround, pathStyle=None):
        # Flying Things and Kitti
        w = args.corpedImgWidth
        h = args.corpedImgHeight

        # get the img, the random crop
        imgL = ReadImg(pathL)
        imgR = ReadImg(pathR)

        if pathStyle != None:
            imgRef = ReadImg(pathStyle)
            imgL, imgR = StyleDataAugmentation(imgL, imgR, imgRef)

        # DataAugmentation
        d = DispDataAugmentation()
        # d = 0
        random_brightness = np.random.uniform(0.5, 2.0, 2)
        random_gamma = np.random.uniform(0.8, 1.2, 2)
        random_contrast = np.random.uniform(0.8, 1.2, 2)

        # random crop
        x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w + d, h)
        # x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
        imgL = ImgSlice(imgL, x, y, w, h)
        imgL = ChromaticTransformations(imgL, random_brightness[0],
                                        random_gamma[0], random_contrast[0])
        imgL = Standardization(imgL)

        # the right img
        imgR = ImgSlice(imgR, x + d, y, w, h)
        # imgR = ImgSlice(imgR, x, y, w, h)
        imgR = ChromaticTransformations(imgR, random_brightness[1],
                                        random_gamma[1], random_contrast[1])
        imgR = Standardization(imgR)

        file_type = os.path.splitext(pathGround)[-1]

        # get groundtrue
        if file_type == ".png":
            imgGround = self.__ReadRandomGroundTrue(pathGround, x, y, w, h)
        else:
            imgGround = self.__ReadRandomPfmGroundTrue(pathGround, x, y, w, h)

        mask = imgGround > 0
        mask = mask.astype(np.float32)
        imgGround = mask * (imgGround + d)

        # imgL, imgR, imgGround, _ = VerticalFlip(
        #    imgL, imgR, imgGround, None)

        imgL = np.expand_dims(imgL, axis=0)
        imgR = np.expand_dims(imgR, axis=0)
        imgGround = np.expand_dims(imgGround, axis=0)

        return imgL, imgR, imgGround

    def __RandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.trainListPath, 2*num+1)
        pathR = GetPath(args.trainListPath, 2*(num + 1))
        pathGround = GetPath(args.trainLabelListPath, num + 1)

        if args.styleTransfer == False:
            pathStyle = None
        else:
            pathStyle = GetPath(args.styleListPath, num % args.styleImgNum + 1)

        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround, pathStyle)

        return imgL, imgR, imgGround

    # Val Flying Things and Kitti
    def __ValRandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.valListPath, 2*num+1)
        pathR = GetPath(args.valListPath, 2*(num + 1))
        pathGround = GetPath(args.valLabelListPath, num + 1)
        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround)

        return imgL, imgR, imgGround

    # Padding Img, used in testing
    def __GetPadingTestData(self, args, num):
        pathL = GetPath(args.testListPath, 2*num+1)
        pathR = GetPath(args.testListPath, 2*(num + 1))

        imgL = ReadImg(pathL)
        imgL = Standardization(imgL)
        imgL = np.expand_dims(imgL, axis=0)

        imgR = ReadImg(pathR)
        imgR = Standardization(imgR)
        imgR = np.expand_dims(imgR, axis=0)

        # pading size
        top_pad = args.padedImgHeight - imgL.shape[1]
        left_pad = args.padedImgWidth - imgL.shape[2]

        # pading
        imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                                 (0, 0)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                                 (0, 0)), mode='constant', constant_values=0)

        name = None
        if args.dataset == "ETH3D" or args.dataset == "Middlebury":
            pos = pathL.rfind('/')
            left_name = pathL[0:pos]
            pos = left_name.rfind('/')
            name = left_name[pos + 1:]

        return imgL, imgR, top_pad, left_pad, name
