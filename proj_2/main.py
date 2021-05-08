# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PyQt5.QtWidgets import QMainWindow
from interface import *
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

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
        self.jacky.pushButton_5.clicked.connect(self.click_1)
        self.jacky.pushButton_2.clicked.connect(self.click_2)
        self.jacky.pushButton_3.clicked.connect(self.click_3)
        self.jacky.pushButton_4.clicked.connect(self.click_4)
        self.jacky.pushButton_12.clicked.connect(self.click_5)
        self.jacky.pushButton_13.clicked.connect(self.click_6)
        self.jacky.pushButton_15.clicked.connect(self.click_7)
        self.jacky.pushButton_17.clicked.connect(self.click_12)
        self.jacky.pushButton_6.clicked.connect(self.click_8)
        self.jacky.pushButton_7.clicked.connect(self.click_9)
        self.jacky.pushButton_14.clicked.connect(self.click_10)
        self.jacky.pushButton_16.clicked.connect(self.click_11)

        self.jacky.btn5_1.clicked.connect(self.click_5_1)
        self.jacky.btn5_2.clicked.connect(self.click_5_2)
        self.jacky.btn5_3.clicked.connect(self.click_5_3)
        self.jacky.btn5_4.clicked.connect(self.click_5_4)
        self.jacky.pushButton_8.clicked.connect(self.click_5_5)
        (self.x, self.y), (self.x_test, self.y_test) = datasets.cifar10.load_data()

    def click_1(self):
        img1 = cv2.imread('image/Uncle_Roger.jpg')
        cv2.imshow('Uncle_Roger', img1)
        print("Height =", img1.shape[0])
        print('Width = ', img1.shape[1])

    def click_2(self):
        img2 = cv2.imread('image/Flower.jpg')
        b,g,r = cv2.split(img2)

        zeros = np.zeros(img2.shape[:2], dtype="uint8")
        merged_b = cv2.merge([b, zeros, zeros])
        merged_g = cv2.merge([zeros, g, zeros])
        merged_r = cv2.merge([zeros, zeros, r])

        cv2.imshow('image', img2)
        cv2.imshow("merged_b", merged_b)
        cv2.imshow("merged_g", merged_g)
        cv2.imshow("merged_r", merged_r)

    def click_3(self):
        img3 = cv2.imread('image/Uncle_Roger.jpg')
        img4 = cv2.flip(img3, 1)
        cv2.imshow('Origin_Uncle', img3)
        cv2.imshow('Flip_Uncle', img4)

    def click_4(self):
        img3 = cv2.imread('image/Uncle_Roger.jpg')
        img3_copy = img3
        img4 = cv2.flip(img3, 1)
        cv2.namedWindow('image')
        def trackbar(pos):
            x = cv2.getTrackbarPos('mix', 'image')
            x = float(x) / 255.0
            img3 = cv2.addWeighted(img3_copy, 1.0 - x, img4, x, 0)
            cv2.imshow('image', img3)
        cv2.createTrackbar('mix', 'image', 0, 255, trackbar)

    def click_5(self):
        img5 = cv2.imread('image/Cat.png')
        cv2.imshow('image', img5)
        median_blur = cv2.blur(img5, (7, 7))
        cv2.imshow('Median', median_blur)

        cv2.waitKey(0)

    def click_6(self):
        img6 = cv2.imread('image/Cat.png')
        cv2.imshow('image', img6)
        gaussian_blur = cv2.GaussianBlur(img6, (3, 3), 0)
        cv2.imshow('Guassian', gaussian_blur)

    def click_7(self):
        img7 = cv2.imread('image/Cat.png')
        cv2.imshow('image', img7)
        bil = cv2.bilateralFilter(img7, 9, 90, 90)
        cv2.imshow('bil', bil)

    def click_8(self):
        ori_img = cv2.imread('image/Chihiro.jpg')
        gray_img = cv2.imread('image/Chihiro.jpg', 0)
        cv2.imshow('ori_img', ori_img)
        cv2.imshow('gray_img', gray_img)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        res = signal.convolve2d(gray_img, gaussian_kernel, boundary='symm', mode='same')
        cv2.imwrite('image/gray_res.png', res)
        gray_res = cv2.imread('image/gray_res.png', 0)
        cv2.imshow('res', gray_res)

    def click_9(self):
        sobx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img9 = cv2.imread('image/gray_res.png')
        gray = cv2.cvtColor(img9, cv2.COLOR_RGB2GRAY)

        arr = np.array(gray)
        new_arr = np.zeros((arr.shape[0], arr.shape[1]))

        for i in range(1, arr.shape[0] - 1):
            for j in range(1, arr.shape[1] - 1):
                t = arr[i - 1:i + 2, j - 1:j + 2]
                a = np.multiply(t, sobx)
                new_arr[i, j] = a.sum()

        new_img = Image.fromarray(new_arr)
        abs_sobel = np.absolute(new_img)
        sobx = np.array(abs_sobel)

        cv2.imwrite('image/sobx.png', sobx)
        img_x = cv2.imread('image/sobx.png')
        cv2.imshow('sobx', img_x)

    def click_10(self):
        soby = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        img10 = cv2.imread('image/gray_res.png')

        gray = cv2.cvtColor(img10, cv2.COLOR_RGB2GRAY)
        arr = np.array(gray)
        new_arr = np.zeros((arr.shape[0], arr.shape[1]))

        for i in range(1, arr.shape[0] - 1):
            for j in range(1, arr.shape[1] - 1):
                t = arr[i - 1:i + 2, j - 1:j + 2]
                a = np.multiply(t, soby)
                new_arr[i, j] = a.sum()

        new_img = Image.fromarray(new_arr)
        abs_sobel = np.absolute(new_img)
        soby = np.array(abs_sobel)

        cv2.imwrite('image/soby.png', soby)
        img_y = cv2.imread('image/soby.png')
        cv2.imshow('soby', img_y)

    def click_11(self):

        img_11_x = cv2.imread('image/sobx.png')
        img_11_y = cv2.imread('image/soby.png')

        gray_x = cv2.cvtColor(img_11_x, cv2.COLOR_RGB2GRAY)
        gray_y = cv2.cvtColor(img_11_y, cv2.COLOR_RGB2GRAY)

        arr_x = np.array(gray_x)
        arr_y = np.array(gray_y)

        sobx = np.absolute(arr_x)
        soby = np.absolute(arr_y)

        new_arr = np.zeros((arr_x.shape[0], arr_x.shape[1]))
        for i in range(1, arr_x.shape[0] - 1):
            for j in range(1, arr_x.shape[1] - 1):
                new_arr[i, j] = np.uint8(np.sqrt(np.square(sobx[i, j] / 2) + np.square(soby[i, j] / 2)))

        new_img = Image.fromarray(new_arr)
        abs_sobel = np.absolute(new_img)

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        cv2.imshow("magnitude", scaled_sobel)

    def click_12(self):
        rot = self.jacky.line1.text()
        sca = self.jacky.line2.text()
        tx = self.jacky.line3.text()
        ty = self.jacky.line4.text()
        img12 = cv2.imread('image/Parrot.png')

        center = (160, 84)

        size = img12.shape
        M = np.array([[1, 0, float(tx)], [0, 1, float(ty)]])
        res = cv2.warpAffine(img12, M, (size[1], size[0]))
        M = cv2.getRotationMatrix2D(center, float(rot), float(sca))
        res = cv2.warpAffine(res, M, (size[1], size[0]))

        cv2.imshow('img12', img12)
        cv2.imshow('res', res)

    def click_5_1(self):
        dic = {0: "airplain", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        self.y = tf.squeeze(self.y, axis=1)
        self.y_test = tf.squeeze(self.y_test, axis=1)
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
        acc = cv2.imread('image/Q5/Accuracy.jpg')
        loss = cv2.imread('image/Q5/Loss.jpg')
        cv2.imshow('acc', acc)
        cv2.imshow('loss', loss)

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

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    print('PyCharm')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
