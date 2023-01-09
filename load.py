# data loading
# train  image x    label y
# test   image x    label y

import struct
import numpy as np
import cv2
# read image  X
def load_image_fromfile(filename):
    with open(filename, 'br') as fd:
        # 读取图像的信息
        header_buf = fd.read(16)   # 16字节，4个int整数
        # 按照字节解析头信息（具体参考python SL的struct帮助）  解包
        magic_, nums_, width_, height_ = struct.unpack('>iiii', header_buf)  # 解析成四个整数：>表示大端字节序，i表示4字节整数
        # 保存成ndarray对象
        imgs_ = np.fromfile(fd, dtype=np.uint8)
        imgs_ = imgs_.reshape(nums_, height_, width_)
    return imgs_


# read labels  Y
def load_label_fromfile(filename):
    with open(filename, 'br') as fd:
        header_buf = fd.read(8) 
        magic, nums = struct.unpack('>ii' ,header_buf) 
        labels_ = np.fromfile(fd, np.uint8) 
    return labels_


train_imgs = load_image_fromfile('第三次实训\\L05\\digit\\data\\train-images.idx3-ubyte')
train_labels = load_label_fromfile('第三次实训\\L05\\digit\\data\\train-labels.idx1-ubyte')
print(train_imgs.shape)
print(train_labels.shape)

test_imgs = load_image_fromfile('第三次实训\\L05\\digit\\data\\t10k-images.idx3-ubyte')
test_labels = load_label_fromfile('第三次实训\\L05\\digit\\data\\t10k-labels.idx1-ubyte')
print(test_imgs.shape)
print(test_labels.shape)

#  train 60000
#  test  10000

#60000 28 28
img1 = train_imgs[50000]
# cv2.imwrite("./第三次实训/L05/digit/my.png",img1)   #cv2不支持中文路径,报错
# cv2.imencode(保存格式, 保存图片)[1].tofile(保存路径)
cv2.imencode('.png',img1)[1].tofile("./第三次实训/L05/digit/my.png")


# background:black
# font      :white