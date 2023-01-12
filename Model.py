#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
from utilities import *

class CNN_2D(nn.Module):

    def __init__(self, in_channel=1, out_channel=8):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,128, kernel_size=(3,3), padding=0),
            nn.ReLU(inplace=True))
            #nn.Dropout (p=0.25))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=0),
            nn.ReLU(inplace=True))
            #nn.Dropout (p=0.25))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), padding=0),
            nn.ReLU(inplace=True))
            #nn.Dropout (p=0.25))
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=0),
            nn.ReLU(inplace=True))
            #nn.Dropout (p=0.25))
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout (p=0.5))
        self.fc = nn.Sequential(
            nn.Linear(16 *2916, 1024))

    def forward(self, x):
        x_conv1 = self.conv1(x)
        self.featuremap_conv1 = x_conv1.detach() # 核心代码
        x_conv2 = self.conv2(x_conv1)
        self.featuremap_conv2 = x_conv2.detach() # 核心代码
        x_conv3 = self.conv3(x_conv2)
        self.featuremap_conv3 = x_conv3.detach() # 核心代码
        x_conv4 = self.conv4(x_conv3)
        self.featuremap_conv4 = x_conv4.detach() # 核心代码
        x_conv5 = self.conv5(x_conv4)
        self.featuremap_conv5 = x_conv5.detach() # 核心代码
        x_conv5 = x_conv5.view(x_conv5.size(0), 16*2916)



        x_main = self.fc(x_conv5)


        
        return x_main
    
    
class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,

            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class CLS_Gear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS_Gear, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1024,256),
            nn.LeakyReLU(),
            nn.Dropout (p=0.25),
            nn.Linear(256,out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, n=5):
        super(Discriminator, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


# In[ ]:


class CLS_0(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS_0, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


# In[ ]:


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x
class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
class Discriminator_1(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_1, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 128),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


class Discriminator_2(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_2, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 512),
	nn.Dropout(0.5),
                nn.Linear(512, 256),
	nn.Dropout(0.5),
                nn.Linear(256, 128),
	nn.Dropout(0.5),
                nn.Linear(128, 32),
	nn.Dropout(0.5),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class Discriminator_3(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_3, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
	nn.Dropout(0.5),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
	nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class Discriminator_4(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_4, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class Discriminator_5(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_5, self).__init__()
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 128),
	nn.Dropout(0.5),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 64),
	nn.Dropout(0.5),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)
