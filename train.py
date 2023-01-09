# data
# train
# save model

import torch
from lenet5 import Lenet5
import load

# train_data
train_x = load.load_image_fromfile('./第三次实训/L05/digit/data/train-images.idx3-ubyte')
train_y = load.load_label_fromfile('./第三次实训/L05/digit/data/train-labels.idx1-ubyte')

# test_data
test_x = load.load_image_fromfile('./第三次实训/L05/digit/data/t10k-images.idx3-ubyte')
test_y = load.load_label_fromfile('./第三次实训/L05/digit/data/t10k-labels.idx1-ubyte')

#####    N*28*28 #####    .view() change dim   #####    N*1*28*28 #####
x = torch.Tensor(train_x).view(train_x.shape[0],1,train_x.shape[1],train_x.shape[2])
y = torch.LongTensor(train_y)  

t_x = torch.Tensor(test_x).view(test_x.shape[0],1,test_x.shape[1],test_x.shape[2])
t_y = torch.LongTensor(test_y)

# DataLoader
train_dataset = torch.utils.data.TensorDataset(x,y)
test_dataset =  torch.utils.data.TensorDataset(t_x,t_y)
#  for in train_loader   2000 batch_size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=2000)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=10000)


model = Lenet5()

epoch = 60

cri = torch.nn.CrossEntropyLoss()
# 创建优化器（指定学习率）
opt = torch.optim.Adam(model.parameters(),lr=0.001)

# epoch
for e in range(epoch):
    # batch
    for data,target in train_loader:
        opt.zero_grad()    # 导数清零

        out = model(data)  # forward()
        loss = cri(out,target)
        loss.backward()
        # update weight
        opt.step()  # 更新权重

    with torch.no_grad():         #这里我们只是想看一下训练效果，不需要用来反向传播更新网络，节约内存
        for data,target in test_loader:
            y_ = model(data)
            y_ = torch.nn.functional.log_softmax(y_,dim=1)
            predict = torch.argmax(y_,dim=1)
            c_rate = (predict==target).float().mean()
            print(F"轮数：{e} ---- 准确率：{c_rate}")
            # save model
    
    # save  trained model
    # if e%100 ==0
    # if c_rate >0.99
    state_dict = model.state_dict()
    torch.save(state_dict,'./第三次实训/L05/digit/data/lenet.pth')  

    