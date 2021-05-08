# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import glob
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QMainWindow

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import load_model

from hw2_4 import *

bgsub_path = 'img/Q1_image/bgSub.mp4'
featureTracking_path = 'img/Q2_image/opticalFlow.mp4'


_PATTERN_SIZE = (11, 8)
_WIDGETS = []
_KEY = lambda x: int(os.path.splitext(x)[0])

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.jacky = Ui_MainWindow()
        self.jacky.setupUi(self)
        self.jacky.Btn_1.clicked.connect(self.click_1_1)
        self.jacky.Btn_2.clicked.connect(self.click_2_1)
        self.jacky.Btn_3.clicked.connect(self.click_2_2)
        self.jacky.Btn_4.clicked.connect(self.click_3_1)
        self.jacky.Btn_5.clicked.connect(self.click_4_1)
        self.jacky.Btn_6.clicked.connect(self.click_4_2)
        self.jacky.Btn5_1.clicked.connect(self.click_5_1)
        self.jacky.Btn5_2.clicked.connect(self.click_5_2)
        self.jacky.Btn5_3.clicked.connect(self.click_5_3)
        self.jacky.Btn5_4.clicked.connect(self.click_5_4)


    def click_1_1(self):
        cap = cv2.VideoCapture(bgsub_path)

        fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(50, 0.9)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        while (1):
            ret, frame = cap.read()

            fgmask_mog = fgbg_mog.apply(frame)
            fgmask_gmg = fgbg_gmg.apply(frame)
            fgmask4 = cv2.morphologyEx(fgmask_gmg, cv2.MORPH_OPEN, kernel, iterations=1)

            cv2.imshow('frame', frame)
            cv2.imshow('fgmask', fgmask_mog)
            # cv2.imshow('gmg',fgmask_gmg)
            # cv2.imshow('MORPH_ELLIPSE',fgmask4)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

    def click_2_1(self):
        cap = cv2.VideoCapture(featureTracking_path)
        _, first_frame = cap.read()
        first_frame = cv2.convertScaleAbs(first_frame)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.84
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(first_frame)

        keyP = cv2.KeyPoint_convert(keypoints)
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # im_with_keypoints = cv2.drawKeypoints(first_frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for x in range(len(keyP)):
            (P_Y, P_X) = keyP[x]
            # print((P_Y, P_X))
            im_rec = cv2.rectangle(first_frame, (int(P_Y - 5), int(P_X - 5)), (int(P_Y + 5), int(P_X + 5)), (0, 0, 255),
                                   1)

        cv2.imshow("rec", im_rec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

    def click_2_2(self):
        lk_params = dict(winSize=(21, 21),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        # color = np.random.randint(0,255,(100,3))
        cap = cv2.VideoCapture(featureTracking_path)
        ret, old_frame = cap.read()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.84
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(old_frame)

        KeyP = np.reshape(cv2.KeyPoint_convert(keypoints), (-1, 1, 2))
        # print(KeyP.shape)
        # print(len(KeyP))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while (1):
            ret, frame = cap.read()

            if not ret:
                break
            im_rec = frame.copy()
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, KeyP, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = KeyP[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
                # circle = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                im_rec = cv2.rectangle(im_rec, (int(a - 5), int(b - 5)), (int(a + 5), int(b + 5)), (0, 0, 255), 1)

            # img = cv2.add(frame,mask)
            imCombine = cv2.add(im_rec, mask)
            if ret == True:
                # video.write(imCombine)
                # cv2.imshow('frame',img)
                cv2.imshow('point', imCombine)
                old_frame = frame.copy()
                KeyP = good_new.reshape(-1, 1, 2)
            else:
                cap.release()
                break

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()
        cap.release()

    def click_3_1(self):
        m_src = cv2.imread('img/Q3_Image/rl.jpg')
        cap = cv2.VideoCapture('img/Q3_Image/test4perspective.mp4')

        while (cap.isOpened()):
            try:
                ret, frame = cap.read()

                if frame is None: break
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                parameters = cv2.aruco.DetectorParameters_create()
                markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(
                    frame, dictionary, parameters=parameters)

                index = np.squeeze(np.where(markerIDs == 25))
                refPt1 = np.squeeze(markerCorners[index[0]])[1]

                index = np.squeeze(np.where(markerIDs == 33))
                refPt2 = np.squeeze(markerCorners[index[0]])[2]

                distance = np.linalg.norm(refPt1 - refPt2)

                scalingFac = 0.02
                pts_dst = [
                    [refPt1[0] - round(scalingFac * distance), refPt1[1] - round((scalingFac * distance))]]
                pts_dst = pts_dst + \
                          [[refPt2[0] + round(scalingFac * distance),
                            refPt2[1] - round(scalingFac * distance)]]

                index = np.squeeze(np.where(markerIDs == 30))
                refPt3 = np.squeeze(markerCorners[index[0]])[0]
                pts_dst = pts_dst + \
                          [[refPt3[0] + round(scalingFac * distance),
                            refPt3[1] + round(scalingFac * distance)]]

                index = np.squeeze(np.where(markerIDs == 23))
                refPt4 = np.squeeze(markerCorners[index[0]])[0]
                pts_dst = pts_dst + \
                          [[refPt4[0] - round(scalingFac * distance),
                            refPt4[1] + round(scalingFac * distance)]]
                pts_src = [[0, 0], [m_src.shape[1], 0], [
                    m_src.shape[1], m_src.shape[0]], [0, m_src.shape[0]]]

                pts_dst = np.float32(pts_dst)
                pts_src = np.float32(pts_src)

                h, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
                im_out = cv2.warpPerspective(m_src, h, (frame.shape[1], frame.shape[0]))

                res = np.where(im_out == 0, frame, im_out)
                res = cv2.resize(res, (700, 500), interpolation=cv2.INTER_CUBIC)
                frame = cv2.resize(frame, (700, 500), interpolation=cv2.INTER_CUBIC)
                result = np.hstack([frame, res])
                cv2.imshow("result", result)
                if (cv2.waitKey(30) & 0xff == ord('q')):
                    break
            except:
                pass

    def click_4_1(self):

        def pca_reconstruction(image, num_features=33):
            """
            This function is equivalent to:
            from sklearn.decomposition import PCA
            pca = PCA(num_features)
            recon = pca.fit_transform(image)
            recon = pca.inverse_transform(recon)
            """
            average_image = np.expand_dims(np.mean(image, axis=1), axis=1)
            X = image - average_image
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            recon = average_image + np.matmul(np.matmul(U[:, :num_features], U[:, :num_features].T), X)
            return np.uint8(np.absolute(recon))

        np.random.seed(42)
        img_path = sorted(glob.glob("img/Q4_Image/*.jpg"))
        gray_img = None
        original_img = None

        for img_test in img_path:
            if gray_img is None:
                img = cv2.imread(img_test)
                original_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
            else:
                img = cv2.imread(img_test)
                img1 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                img2 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
                original_img = np.concatenate((original_img, img1), axis=0)
                gray_img = np.concatenate((gray_img, img2), axis=0)

        img_shape = original_img.shape
        r = original_img[:, :, :, 0].reshape(img_shape[0], -1)
        g = original_img[:, :, :, 1].reshape(img_shape[0], -1)
        b = original_img[:, :, :, 2].reshape(img_shape[0], -1)
        r_r, r_g, r_b = pca_reconstruction(r), pca_reconstruction(g), pca_reconstruction(b)
        recon_img = np.dstack((r_r, r_g, r_b))
        recon_img = np.reshape(recon_img, img_shape)

        # Setup a figure 6 inches by 6 inches
        fig = plt.figure(figsize=(17, 4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        length = 17
        for i in range(length):
            ax1 = fig.add_subplot(4, length, i + 1, xticks=[], yticks=[])
            ax1.imshow(original_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax2 = fig.add_subplot(4, length, i + length + 1, xticks=[], yticks=[])
            ax2.imshow(recon_img[i], cmap=plt.cm.bone, interpolation='nearest')
            ax3 = fig.add_subplot(4, length, i + length * 2 + 1, xticks=[], yticks=[])
            ax3.imshow(original_img[i + length], cmap=plt.cm.bone, interpolation='nearest')
            ax4 = fig.add_subplot(4, length, i + length * 3 + 1, xticks=[], yticks=[])
            ax4.imshow(recon_img[i + length], cmap=plt.cm.bone, interpolation='nearest')

        plt.show()

    def click_4_2(self):
        def pca_reconstruction(image, num_features=33):
            """
            This function is equivalent to:
            from sklearn.decomposition import PCA
            pca = PCA(num_features)
            recon = pca.fit_transform(image)
            recon = pca.inverse_transform(recon)
            """
            average_image = np.expand_dims(np.mean(image, axis=1), axis=1)
            X = image - average_image
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            recon = average_image + np.matmul(np.matmul(U[:, :num_features], U[:, :num_features].T), X)
            return np.uint8(np.absolute(recon))

        def reconstruction_error(gray):
            gray = gray.reshape(gray.shape[0], -1)
            gray_recon = pca_reconstruction(gray)
            re = np.sum(np.abs(gray - gray_recon), axis=1)
            return np.mean(re)
        np.random.seed(42)
        gray_img = None
        original_img = None
        for i in range(1, 35):
            if gray_img is None:
                img = cv2.imread('img/Q4_Image\\' + str(i) + '.jpg')
                original_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
            else:
                img = cv2.imread('img/Q4_Image\\' + str(i) + '.jpg')
                img1 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
                img2 = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0)
                original_img = np.concatenate((original_img, img1), axis=0)
                gray_img = np.concatenate((gray_img, img2), axis=0)

        total_error = []
        for img in gray_img:
            error = reconstruction_error(img)
            total_error.append(error)
        print("Reconstruction Error:", total_error)

    def click_5_1(self):
        # print("[001/015] 255.81 sec(s) Train Acc: 0.588212 Loss: 0.042064 | Val Acc: 0.644800 loss: 0.039310")
        # print("[002/015] 261.49 sec(s) Train Acc: 0.677379 Loss: 0.037449 | Val Acc: 0.707200 loss: 0.036967")
        # print("[003/015] 262.77 sec(s) Train Acc: 0.732675 Loss: 0.033910 | Val Acc: 0.754400 loss: 0.030774")
        # print("[004/015] 263.44 sec(s) Train Acc: 0.761435 Loss: 0.030856 | Val Acc: 0.807200 loss: 0.026251")
        # print("[005/015] 262.79 sec(s) Train Acc: 0.798551 Loss: 0.027394 | Val Acc: 0.708800 loss: 0.040540")
        # print("[006/015] 262.89 sec(s) Train Acc: 0.821176 Loss: 0.025098 | Val Acc: 0.831200 loss: 0.024857")
        # print("[007/015] 263.23 sec(s) Train Acc: 0.838912 Loss: 0.022884 | Val Acc: 0.900000 loss: 0.015484")
        # print("[008/015] 262.87 sec(s) Train Acc: 0.853936 Loss: 0.020625 | Val Acc: 0.841200 loss: 0.023092")
        # print("[009/015] 263.05 sec(s) Train Acc: 0.866693 Loss: 0.018931 | Val Acc: 0.921600 loss: 0.012802")
        # print("[010/015] 263.13 sec(s) Train Acc: 0.881984 Loss: 0.017347 | Val Acc: 0.894400 loss: 0.017913")
        # print("[011/015] 263.00 sec(s) Train Acc: 0.892074 Loss: 0.016163 | Val Acc: 0.922800 loss: 0.012087")
        # print("[012/015] 262.88 sec(s) Train Acc: 0.899498 Loss: 0.015189 | Val Acc: 0.941600 loss: 0.009736")
        # print("[013/015] 263.31 sec(s) Train Acc: 0.905454 Loss: 0.014267 | Val Acc: 0.887200 loss: 0.018436")
        # print("[014/015] 262.95 sec(s) Train Acc: 0.911366 Loss: 0.013362 | Val Acc: 0.928400 loss: 0.010985")
        # print("[015/015] 262.78 sec(s) Train Acc: 0.911055 Loss: 0.013047 | Val Acc: 0.927200 loss: 0.010813")
        print("[001/005] 104.05 sec(s) Train Acc: 0.591138 Loss: 0.042446 | Val Acc: 0.669613 loss: 0.038570")
        print("[002/005] 103.24 sec(s) Train Acc: 0.673622 Loss: 0.037993 | Val Acc: 0.692818 loss: 0.037700")
        print("[003/005] 103.32 sec(s) Train Acc: 0.719774 Loss: 0.035279 | Val Acc: 0.720442 loss: 0.038500")
        print("[004/005] 103.27 sec(s) Train Acc: 0.741377 Loss: 0.033147 | Val Acc: 0.737017 loss: 0.032527")
        print("[005/005] 103.33 sec(s) Train Acc: 0.765681 Loss: 0.031379 | Val Acc: 0.771271 loss: 0.031345")

    def click_5_2(self):
        img = cv2.imread('img/Q5_Image/tensorboard.PNG',)
        cv2.imshow('img', img)

    def click_5_3(self):
        r = random.randint(1, 1000)
        #print(r)
        test = cv2.imread('img/Q5_Image/test/'+str(r)+'.jpg')
        saved_model = load_model("ResNet50_10_origin_rgb_10000.h5")
        test= cv2.resize(test,(224, 224))
        test = test.reshape(-1, 224, 224, 3)
        predict = saved_model.predict(test)
        print(predict)
        test = test.reshape(224,224, 3)

        if predict[0][0] < predict[0][1]:
            plt.title("Class:dog")
        else:
            plt.title("Class:cat")
        plt.imshow(test)
        plt.show()


    def click_5_4(self):
        acc = [17.127, 25.6]
        candidates = ["Before random_erase", "After random_erase"]
        x = np.arange(len(candidates))
        plt.bar(x, acc, tick_label=candidates, width=0.5, bottom=60)
        plt.title('Random-Erasing Algo')  # 設定圖形標題
        #plt.xlabel('Candidates')  # 設定X軸標籤
        plt.ylabel('accuracy(%)')  # 設定Y軸標籤
        plt.show()

        #img = cv2.imread("erase_cho.png")
        #cv2.imshow("img",img)




if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
