import torch
from lenet5 import Lenet5
import numpy as np

model = Lenet5()
state = torch.load('./第三次实训/L05/digit/data/lenet.pth')
model.load_state_dict(state)

# ----------use-------------
# mode    1*1*28*28

import cv2
# img = cv2.imread('./第三次实训/L05/digit/img.png')   # 中文路径会报错
img= cv2.imdecode(np.fromfile('./第三次实训/L05/digit/img.png', dtype=np.uint8), -1)
# imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  #cv2.imread读取的图片效果一致

# if else
# img = 255 - img


# print(img.shape)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = cv2.resize(img,(28,28))   #resize

img = torch.Tensor(img).view(1,1,img.shape[0],img.shape[1])
# 1 * 1 * 28 *28

y_ = model(img)  #N[10]
y_ = torch.nn.functional.log_softmax(y_,dim=0)
predict = torch.argmax(y_,dim=0)
print(predict.numpy())

