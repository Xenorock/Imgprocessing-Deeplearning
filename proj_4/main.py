# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from PyQt5.QtWidgets import QMainWindow
from matplotlib import pyplot as plt
from hw2_5 import *
import sys
import cv2 as cv
import numpy as np
import glob
import os
import random

from tensorflow.keras.models import load_model
_KEY = lambda x: int(os.path.splitext(x)[0])

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.jacky = Ui_MainWindow()
        self.jacky.setupUi(self)
        self.jacky.btn_1_1.clicked.connect(self.click_1_1)
        self.jacky.btn_1_2.clicked.connect(self.click_1_2)
        self.jacky.btn_2_1.clicked.connect(self.click_2_1)
        self.jacky.btn_2_2.clicked.connect(self.click_2_2)
        self.jacky.btn_2_3.clicked.connect(self.click_2_3)
        self.jacky.btn_2_4.clicked.connect(self.click_2_4)
        self.jacky.btn_3_1.clicked.connect(self.click_3_1)
        self.jacky.btn_4_1.clicked.connect(self.click_4_1)
        self.jacky.btn_5_1.clicked.connect(self.click_5_1)
        self.jacky.btn_5_2.clicked.connect(self.click_5_2)
        self.jacky.btn_5_3.clicked.connect(self.click_5_3)
        self.jacky.btn_5_4.clicked.connect(self.click_5_4)


        self.jacky.comboBox.addItems(sorted(os.listdir('img/Q2_Image'), key=_KEY))


    def click_1_1(self):
        img1 = cv.imread('img/Q1_Image/coin01.jpg')
        img2 = cv.imread('img/Q1_Image/coin02.jpg')

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        blur1 = cv.GaussianBlur(gray1, (11, 11), 0)
        blur2 = cv.GaussianBlur(gray2, (11, 11), 0)
        binary1 = cv.Canny(blur1, 20, 160)
        binary2 = cv.Canny(blur2, 20, 160)

        (_, contours1, hierarchy1) = cv.findContours(binary1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        (_,contours2, hierarchy2) = cv.findContours(binary2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(img1, contours1, -1, (0, 0, 255), 2)
        cv.drawContours(img2, contours2, -1, (0, 0, 255), 2)

        cv.imshow('win1', img1)
        cv.imshow('win2', img2)


    def click_1_2(self):
        img1 = cv.imread('img/Q1_Image/coin01.jpg')
        img2 = cv.imread('img/Q1_Image/coin02.jpg')

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        blur1 = cv.GaussianBlur(gray1, (11, 11), 0)
        blur2 = cv.GaussianBlur(gray2, (11, 11), 0)
        binary1 = cv.Canny(blur1, 20, 160)
        binary2 = cv.Canny(blur2, 20, 160)

        (_, contours1, hierarchy1) = cv.findContours(binary1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        (_, contours2, hierarchy1) = cv.findContours(binary2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        count1 = 0
        count2 = 0
        for c in contours1:
            count1 += 1
        for c in contours2:
            count2 += 1
        self.jacky.label_1.setText("There are " + str(count1) + " coins in coin01.jpg")
        self.jacky.label_2.setText("There are " + str(count2) + " coins in coin02.jpg")


    def click_2_1(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        objpoints = []  # 3d
        imgpoints = []  # 2d
        images = glob.glob('img/Q2_Image/*.bmp')
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

    def click_2_2(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q2_Image/*.bmp')
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

    def click_2_3(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q2_Image/*.bmp')
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
        i = self.jacky.comboBox.currentIndex()
        R, _ = cv.Rodrigues(rvecs[i])

        print(np.hstack([R, tvecs[i]]))

    def click_2_4(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q2_Image/*.bmp')

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

    def click_3_1(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11:, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('img/Q3_Image/*.bmp')
        # a = '1'
        arr = []
        for i in range(1, 6):
            t = cv.imread('img/Q3_Image/' + str(i) + '.bmp')
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


    def click_4_1(self):
        # imgL = cv.imread('img/Q4_Image/imgL.png', 0)
        # imgR = cv.imread('img/Q4_Image/imgR.png', 0)
        # stereo = cv.StereoBM_create(numDisparities=160, blockSize=15)
        # disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16
        # disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        #
        # a = cv.resize(disparity, (1200, 800))
        # cv.imshow('123', a)

        imgL = cv.imread('img/Q4_Image/imgL.png', 0)
        imgR = cv.imread('img/Q4_Image/imgR.png', 0)

        minDisp = 5
        numDisp = 160 - minDisp
        windowSize = 1

        stereo = cv.StereoBM_create(
            numDisparities=160,
            blockSize=15
        )

        disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16
        disparity = (disparity - minDisp) / numDisp

        # print(disparity)

        disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        print(disparity)
        print(disparity.shape)

        img_resize = cv.resize(disparity, (1200, 800))
        cv.imshow("image", img_resize)

        a = []
        b = []

        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (x, y)
                a.append(x)
                b.append(y)
                color = (255, 255, 0)
                thickness = 2
                cv.circle(img_resize, (x, y), 1, color, thickness=3)
                # cv2.putText(img_resize, xy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness)
                cv.imshow("image", img_resize)
                # print(x, y)

                print("Disparity: " + str(round(disparity[y][x], 2)))

                if disparity[y][x] != 0:
                    depth = (178 * 2826) / disparity[y][x]
                    print("depth: " + str(int(depth)))
                else:
                    print("inf")

        cv.namedWindow("image")
        cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        # cv2.imshow("image", img_resize)
        cv.waitKey(0)
        # print(a[0], b[0])

        cv.waitKey()
        cv.destroyAllWindows()
    def click_5_1(self):
        print("[001/005] 104.05 sec(s) Train Acc: 0.591138 Loss: 0.042446 | Val Acc: 0.669613 loss: 0.038570")
        print("[002/005] 103.24 sec(s) Train Acc: 0.673622 Loss: 0.037993 | Val Acc: 0.692818 loss: 0.037700")
        print("[003/005] 103.32 sec(s) Train Acc: 0.719774 Loss: 0.035279 | Val Acc: 0.720442 loss: 0.038500")
        print("[004/005] 103.27 sec(s) Train Acc: 0.741377 Loss: 0.033147 | Val Acc: 0.737017 loss: 0.032527")
        print("[005/005] 103.33 sec(s) Train Acc: 0.765681 Loss: 0.031379 | Val Acc: 0.771271 loss: 0.031345")

    def click_5_2(self):
        img = cv.imread('img/Q5_Image/tensorboard.PNG',)
        cv.imshow('img', img)

    def click_5_3(self):
        r = random.randint(1, 1000)
        #print(r)
        test = cv.imread('img/Q5_Image/test/'+str(r)+'.jpg')
        saved_model = load_model("ResNet50_10_origin_rgb_10000.h5")
        test= cv.resize(test,(224, 224))
        test = test.reshape(-1, 224, 224, 3)
        predict = saved_model.predict(test)
        print(predict)
        test = test.reshape(224, 224, 3)

        if predict[0][0] < predict[0][1]:
            cv.namedWindow('Class:dog')
            test = cv.resize(test, (600, 600))
            cv.imshow('Class:dog',test)
            cv.resizeWindow('Class:dog',600, 600)
        else:
            cv.namedWindow('Class:cat')
            test = cv.resize(test, (600, 600))
            cv.imshow('Class:cat',test)
            cv.resizeWindow('Class:cat',600, 600)


    def click_5_4(self):
        acc = [17.127, 20.623]
        candidates = ["Before Resize", "After Resize"]
        x = np.arange(len(candidates))
        plt.bar(x, acc, tick_label=candidates, width=0.5, bottom=60, color='rgb')
        plt.title('Resize Augmentation Comparison')  # 設定圖形標題
        #plt.xlabel('Candidates')  # 設定X軸標籤
        plt.ylabel('accuracy(%)')  # 設定Y軸標籤
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
