# ----Lenet5  CNN   calculation ------ 
# 3层卷积和2层全连接

import torch

# layer

class Lenet5(torch.nn.Module):
    # constructor function
    def __init__(self):
        super(Lenet5,self).__init__()
        # init data
        # init layer

        #N*1*32*32   5*5    N*6*28*28
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),padding=2)
    
        # N*6*28*28  N*6*14*14   N*16*10*10
        self.layer2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),padding=0)

        # N*16*10*10 N*16*5*5     N*120*1*1  
        self.layer3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5),padding=0)

        #  N*120*1*1    N*84
        self.layer4 = torch.nn.Linear(120,84)  #通常用于设置网络中的全连接层

        #  N*84    N*10
        self.layer5 = torch.nn.Linear(84,10)



    # 重写 父类的方法  calculation
    def forward(self,input):
        #   layer1
        o = self.layer1(input)                                # N*1*32*32====>N*6*28*28
        o = torch.nn.functional.relu(o)                         
        o = torch.nn.functional.max_pool2d(o,kernel_size=(2,2))   # N*6*28*28====>N*6*14*14

        # layer2
        o = self.layer2(o)                                     # N*6*14*14====>N*16*10*10
        o = torch.nn.functional.relu(o) 
        o = torch.nn.functional.max_pool2d(o,kernel_size=(2,2))   # N*16*10*100====>N*16*5*5

        # layer03
        o = self.layer3(o)                                     # N*16*5*5====>N*120*1*1                           
        o = torch.nn.functional.relu(o) 
        o = o.squeeze()                                        # N*120*1*1====>N*120

        # layer04
        o = self.layer4(o)                                     # N*120====>N*84
        o = torch.nn.functional.relu(o) 

         # layer05
        o = self.layer5(o)                                      # N*84====>N*10

        return o 







    # def calculation(self,input):
    #     # calculation
    #     # torch.nn.fuctionnal.con2d()   pood2d()   linear()





# net = Lenet5()
# y_ = net(input)  # foraward()

