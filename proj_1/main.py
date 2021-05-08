# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from PyQt5.QtWidgets import QMainWindow
from matplotlib import pyplot as plt
from hw1 import *
import sys
import numpy as np
import cv2 as cv

import glob
import os
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import load_model


_KEY = lambda x: int(os.path.splitext(x)[0])

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.jacky = Ui_MainWindow()
        self.jacky.setupUi(self)
        self.jacky.but1.clicked.connect(self.click_1)
        self.jacky.but2.clicked.connect(self.click_2)
        self.jacky.but3.clicked.connect(self.click_3)
        self.jacky.but4.clicked.connect(self.click_4)
        self.jacky.but2_1.clicked.connect(self.click_2_1)
        self.jacky.but3_1.clicked.connect(self.click_3_1)
        self.jacky.but4_1.clicked.connect(self.click_4_1)
        self.jacky.but4_2.clicked.connect(self.click_4_2)
        self.jacky.box.addItems(sorted(os.listdir('img/Q1'), key=_KEY))

        self.jacky.btn5_1.clicked.connect(self.click_5_1)
        self.jacky.btn5_2.clicked.connect(self.click_5_2)
        self.jacky.btn5_3.clicked.connect(self.click_5_3)
        self.jacky.btn5_4.clicked.connect(self.click_5_4)
        self.jacky.pushButton_5.clicked.connect(self.click_5_5)
        (self.x, self.y), (self.x_test, self.y_test) = datasets.cifar10.load_data()

    def click_1(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        objpoints = []  # 3d
        imgpoints = []  # 2d
        images = glob.glob('img/Q1/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            cv.namedWindow('img', 0)
            cv.resizeWindow('img', 1000, 1000)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (11, 8), corners, ret)
                cv.imshow('img', img)
                cv.waitKey(300)
        cv.destroyAllWindows()

    def click_2(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q1/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (11, 8), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx)

    def click_3(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q1/*.bmp')
        # a = '1'
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv.drawChessboardCorners(img, (11, 8), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        i = self.jacky.box.currentIndex()
        R, _ = cv.Rodrigues(rvecs[i])

        print(np.hstack([R, tvecs[i]]))

    def click_4(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q1/*.bmp')
        # a = '1'
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv.drawChessboardCorners(img, (11, 8), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(dist)

    def click_2_1(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q2/*.bmp')
        # a = '1'
        arr=[]
        for i in range(1,6):
            t=cv.imread('img/Q2/'+str(i)+'.bmp')
            arr.append(t)
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv.drawChessboardCorners(img, (11, 8), corners, ret)
        cv.namedWindow('win', 0)
        cv.resizeWindow('win', (1000, 1000))
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        for i in range(5):
            corners = np.array([(3, 3, -3), (1, 1, 0), (3, 5, 0), (5, 1, 0), ], dtype=np.float32)
            corners, _ = cv.projectPoints(corners, rvecs[i], tvecs[i], mtx, dist)
            corners = np.squeeze(corners, axis=1)
            corners = [tuple(c) for c in corners]

            cv.line(arr[i], corners[0], corners[1], [0, 0, 255], 10)
            cv.line(arr[i], corners[0], corners[2], [0, 0, 255], 10)
            cv.line(arr[i], corners[0], corners[3], [0, 0, 255], 10)
            cv.line(arr[i], corners[1], corners[2], [0, 0, 255], 10)
            cv.line(arr[i], corners[1], corners[3], [0, 0, 255], 10)
            cv.line(arr[i], corners[2], corners[3], [0, 0, 255], 10)
            cv.imshow('win', arr[i])
            cv.waitKey(500)
        cv.destroyAllWindows()


    def click_3_1(self):
        imgL = cv.imread('img/Q3/imL.png', 0)
        imgR = cv.imread('img/Q3/imR.png', 0)
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        cv.StereoBM_create()
        plt.imshow(disparity, 'gray')
        plt.show()

    def click_4_1(self):
        img1 = cv.imread('img/Q4/Aerial1.jpg')
        img2 = cv.imread('img/Q4/Aerial2.jpg')

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp1 = sift.detect(gray1, None)
        kp2 = sift.detect(gray2, None)

        sort1 = sorted(kp1, key=lambda kp1: kp1.size, reverse=True)
        sort2 = sorted(kp2, key=lambda kp2: kp2.size, reverse=True)
        img1 = cv.drawKeypoints(gray1, sort1[: 7], img1)
        img2 = cv.drawKeypoints(gray2, sort2[: 7], img2)

        #cv.imshow('img1', img1)
        #cv.imshow('img2', img2)
        fig1 = np.hstack((img1, img2))
        cv.imshow('figure 1', fig1)

        cv.imwrite('img/Q4/FeatureAerial1.jpg', img1)
        cv.imwrite('img/Q4/FeatureAerial2.jpg', img2)

    def click_4_2(self):
        img1 = cv.imread('img/Q4/Aerial1.jpg')
        img2 = cv.imread('img/Q4/Aerial2.jpg')

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp1 = sift.detect(gray1, None)
        kp2 = sift.detect(gray2, None)

        sort1 = sorted(kp1, key=lambda kp1: kp1.size, reverse=True)
        sort2 = sorted(kp2, key=lambda kp2: kp2.size, reverse=True)

        des1 = sift.compute(img1, sort1[: 7])
        des2 = sift.compute(img2, sort2[: 7])

        img1 = cv.drawKeypoints(gray1, sort1[: 7], img1)
        img2 = cv.drawKeypoints(gray2, sort2[: 7], img2)

        match = cv.BFMatcher()
        matches = match.knnMatch(des1[1], des2[1], k=2)

        app = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                app.append([m])

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(0, 0, 255),
                           flags=0)

        res = cv.drawMatchesKnn(img1, sort1[:7], img2, sort2[:7], app, None, **draw_params)
        cv.imshow("result", res)
        cv.waitKey()
        cv.destroyAllWindows()

    def click_5_1(self):
        dic = {0: "airplain", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        self.y = tf.squeeze(self.y, axis=1)
        self.y_test = tf.squeeze(self.y_test, axis=1)
        '''cv.namedWindow('img', 0)
        cv.resizeWindow('img', 500, 500)
        n = 1
        for i in range(10):
            i = random.randint(0, 9999)
            cv.imshow('img', x[i])
            #print(dic[int(y[i])])
            print('Label ' + str(n)+': ' + dic[int(y[i])])
            n = n + 1
            cv.waitKey(500)
        cv.destroyAllWindows()'''
        for i in range(10):
            r = random.randint(0, 9999)
            plt.subplot(2, 5, i+1)
            plt.title(dic[int(self.y[r])])
            plt.imshow(self.x[r])
            plt.axis('off')
        plt.show()

    def click_5_2(self):
        print('hyperparameters: ' + '\n' + 'batch size: ' + str(32) + '\n' + 'learning rate: ' + str(0.001) + '\n' + 'optimizer: ADAM')

    def click_5_3(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=10, activation="softmax"))

        model.summary()
    def click_5_4(self):
        acc = cv.imread('img/Q5/Accuracy.jpg')
        loss = cv.imread('img/Q5/Loss.jpg')
        cv.imshow('acc', acc)
        cv.imshow('loss', loss)

    def click_5_5(self):
        '''val = self.lineEdit.text()
        print(val)
        img1 = img[val].reshape(-1,32,32,3)
        '''
        saved_model = load_model("model.h5")
        index = self.jacky.lineEdit.text()
        if int(index) < 0 or int(index) > 9999:
            print("Error Test Image Index. Please enter number in range 0 to 9999")
        else:
            img_predict = self.x_test[int(index)]
            plt.figure(num="Test Image", figsize=(10, 9))
            plt.imshow(self.x_test[int(index)])
            plt.axis("off")

            img_predict = img_predict.reshape(-1, 32, 32, 3)
            predict = saved_model.predict(img_predict)
            # 設定浮點數格式
            float_formatter = lambda x: "%.7f" % x
            np.set_printoptions(formatter={'float_kind': float_formatter})

            print(predict[0])
            y = []
            for i in range(10):
                   y.append(predict[0][i])

            plt.figure(num='Test', figsize=(10, 9))
            plt.bar(range(10), y, color='gold')
            plt.xticks(range(10), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
            plt.xticks(rotation='vertical')
            plt.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



