
import os
import torch
from PIL import Image
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import sample
from PIL import Image, ImageChops, ImageEnhance
import random

class DeepFakeDataset(Dataset):

    def __init__(self, path, set = 'train', transform = None):
        self.path = path

        self.transform = transform

        if set == 'train' or set == 'val' or set == 'test':
            self.fake_path = path + '/' + set + '/fake'
            self.real_path = path + '/' + set + '/real' 
        else:
            assert set == 'train' or set == 'val' or set == 'test', 'set name must be train, val, test'

        self.real_list = glob(os.path.join(self.real_path, '*.jpg'))
        self.fake_list = glob(os.path.join(self.fake_path, '*.jpg'))

        # fake 1, real 0
        self.img_list = self.real_list + self.fake_list
        self.target_list = [0]*len(self.real_list) + [1]*len(self.fake_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        target = self.target_list[idx]

        data = Image.open(img_path).convert('RGB')

        if not self.transform == None:
            data = self.transform(data)
        
        return data, target

class ImageNetDatset(Dataset):

    def __init__(self, path, count = 0, set = 'train', transform = None):
        self.path = path

        self.transform = transform
        
        if set == 'train' or set == 'val' or set == 'test':
            self.file_path = path + '/' + set
        else:
            assert set == 'train' or set == 'val' or set == 'test', 'set name must be train, val, test'

        self.img_list = glob(os.path.join(self.file_path, '**/*.JPEG'))
        if count != 0:
            self.img_list = sample(self.img_list, count)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        data = Image.open(img_path).convert('RGB')

        if not self.transform == None:
            data = self.transform(data)
        
        return data

class TestDataset(Dataset):

    def __init__(self, path = './test', transform = None):
        self.path = path

        self.transform = transform

        
        self.fake_path = path + '/input' + '/fake'
        self.real_path = path + '/input' + '/real' 
        

        self.real_list = glob(os.path.join(self.real_path, '*.jpg'))
        self.fake_list = glob(os.path.join(self.fake_path, '*.jpg'))

        # fake 1, real 0
        self.img_list = self.real_list + self.fake_list
        self.target_list = [0]*len(self.real_list) + [1]*len(self.fake_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        target = self.target_list[idx]

        data = Image.open(img_path).convert('RGB')

        if not self.transform == None:
            data = self.transform(data)
        
        return data, target



class ELA(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, x):
        temp_filename = 'tmp.jpg'
        x = x.resize(self.output_size)
        x.save(temp_filename, 'JPEG', quality = 75)
        temp_image = Image.open(temp_filename)
        ela_image = ImageChops.difference(x, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
        return ela_image
    
if __name__ == "__main__":
    data_path = "D:\BoB\심화교육\오동빈\과제\ai_eraser\DeepErase"

    # transform = transforms.Compose([
    #         transforms.Resize((299, 299)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ])
    # transform = transforms.Compose([
    #     ELA(),
    #     transforms.Resize((299, 299)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    # dataset = DeepFakeDatset(data_path, set = 'train', transform = transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)
    
    # for data, target in dataloader:
    #     data = data.squeeze(0)
    #     print(data.shape)
    #     data = data.detach().cpu().numpy().transpose((1,2,0))*255
    #     data = data.astype(np.uint8)
    #     im = Image.fromarray(data)
    #     im.show()
    #     print(data.shape)
    #     print(target)
    #     break
    #     #im.show()
        
    transform = transforms.Compose([
        ELA((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = TestDataset(transform = transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)
    
    for data in dataloader:
        data = data.squeeze(0)
        print(data.shape)
        data = data.detach().cpu().numpy().transpose((1,2,0))*255
        data = data.astype(np.uint8)
        im = Image.fromarray(data)
        im.show()
        print(data.shape)
        
        #im.show()