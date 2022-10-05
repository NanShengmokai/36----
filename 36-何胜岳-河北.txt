##############  1实现RGB2Gray######################

# 用PIL中Image做图片的RGB2GRAY，有噪声

from PIL import Image
image = Image.open('./timg3.jpg')  #打开图片
image = image.convert('1')    #将图片转化为黑白图像
image.save('timg4.jpg')     #保存图片

from PIL import Image
image=Image.open('timg3.jpg')
image=image.convert('1')
image.save('timg6.jpg')

# 用CV2做图片的RGB2GRAY

import cv2   #导入opencv
img = cv2.imread('timg3.jpg')   #图片导入
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将导入的图片进行灰度化处理
cv2.imwrite('./timg5.jpg',gray)    #imwrite()函数用来保存图片

import cv2
img=cv2.imread('timg3.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('timg7.jpg',gray)


################ 2实现图片的二值化########################
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文

# 读取灰度图像
img = cv2.imread("timg3.jpg",0)
print("原图的shape: ", img.shape)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("灰度值图片")
# plt.show()


# 1.全局阈值法
ret, mask_all = cv2.threshold(src=img,                  # 要二值化的图片
                              thresh=127,               # 全局阈值
                              maxval=255,               # 大于全局阈值后设定的值
                              type=cv2.THRESH_BINARY)   # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
print("全局阈值的shape: ", mask_all.shape)
plt.subplot(2, 2, 2)
plt.imshow(mask_all, cmap='gray')
plt.title("全局阈值")
# plt.show()


# 2.自适应阈值法
mask_local = cv2.adaptiveThreshold(src=img,                                     # 要进行处理的图片
                                   maxValue=255,                                # 大于阈值后设定的值
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,   # 自适应方法，ADAPTIVE_THRESH_MEAN_C：表区域内均值；ADAPTIVE_THRESH_GAUSSIAN_C：表区域内像素点加权求和
                                   thresholdType=cv2.THRESH_BINARY,             # 同全局阈值法中的参数一样
                                   blockSize=11,                                # 方阵（区域）大小，
                                   C=1)                                         # 常数项，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
print("局部阈值的shape: ", mask_local.shape)
plt.subplot(2, 2, 3)
plt.imshow(mask_local, cmap='gray')
plt.title("局部阈值")

# 3.OTSU二值化
ret2, mask_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("OTSU的shape: ", mask_OTSU.shape)
plt.subplot(2, 2, 4)
plt.imshow(mask_OTSU, cmap='gray')
plt.title("二值化")

plt.show()
