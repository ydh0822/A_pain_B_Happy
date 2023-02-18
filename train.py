import torch

from model import *

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops, ImageEnhance
from score_utils import *
from data import *
import random


transform = transforms.Compose([
    ELA((299,299)),
    #transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
])

data_path = 'D:\BoB\심화교육\오동빈\과제\ai_eraser\DeepErase'
trainset = DeepFakeDataset(data_path, set = 'train', transform = transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=False)

validset = DeepFakeDataset(data_path, set = 'val', transform = transform)
validloader = DataLoader(validset, batch_size=8, shuffle=True, drop_last=False)

#hyper parameter
epochs = 50
lr = 0.002

#model
model = Xception(num_classes=2).cuda()

#loss function
criterion = nn.CrossEntropyLoss().cuda()

#optimizer
optimizer = optim.Adam(model.parameters(), lr= lr)
schedular = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.2)

for epoch in range(epochs):
    #train
    model.train()
    train_bar = tqdm(trainloader)
    train_cost = AverageMeter()
    train_acc = AverageMeter()
    for data, target in train_bar:
        
        #gpu load
        data = data.cuda()
        target = target.cuda()
        
        #gradient init
        optimizer.zero_grad()
        
        #predict
        pred = model(data)
        
        #loss calc
        loss = criterion(pred, target)
        
        #gradient stack
        loss.backward()
        
        #gradient apply
        optimizer.step()

        num_data = len(data)
        train_cost.update(loss.item(), num_data)
        train_acc.update(Accuracy(pred, target), num_data)
    
        train_bar.set_description(desc = '[%d/%d]   cost: %.9f     acc: %.9f %%' % (
            epoch+1, epochs, train_cost.avg, train_acc.avg
        ))

    print('[Train][Epoch : {:>3}   cost = {:>.9}]'.format(epoch+1, train_cost.avg))
    
    #validation
    model.eval()
    val_cost = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():

        val_bar = tqdm(validloader)
        for data, target in val_bar:

            data = data.cuda()
            target = target.cuda()

            pred = model(data)

            loss = criterion(pred, target)

            num_data = len(data)
            val_cost.update(loss.item(), num_data)
            val_acc.update(Accuracy(pred, target), num_data)

            val_bar.set_description(desc = '[%d/%d]   cost: %.9f     acc: %.9f %%' % (
            epoch+1, epochs, val_cost.avg, val_acc.avg
            ))
        print('[Valid][Epoch : {:>3}   cost = {:>.9}]'.format(epoch+1, val_cost.avg))
    output_path = './checkpoints/'
    torch.save(model.state_dict(), output_path + "xception_" + str(epoch+1) + ".pt")



    # 47

    