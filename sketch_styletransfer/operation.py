import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import subprocess as sp
from PIL import Image
import time
import torch.utils.data as data



### model math functions

from skimage.color import hsv2rgb
import torch.nn.functional as F
import torch.nn as nn

eps = 1e-7
class HSV_Loss(nn.Module):
    def __init__(self, h=0, s=1, v=0.7):
        super(HSV_Loss, self).__init__()
        self.hsv = [h, s, v]
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    @staticmethod
    def get_h(im):
        img = im * 0.5 + 0.5
        b, c, h, w = img.shape
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)
        hue[img[:,2]==img.max(1)[0]] = 4.0+((img[:,0]-img[:,1])/(img.max(1)[0] - img.min(1)[0]))[img[:,2]==img.max(1)[0]]
        hue[img[:,1]==img.max(1)[0]] = 2.0+((img[:,2]-img[:,0])/(img.max(1)[0] - img.min(1)[0]))[img[:,1]==img.max(1)[0]]
        hue[img[:,0]==img.max(1)[0]] = ((img[:,1]-img[:,2])/(img.max(1)[0] - img.min(1)[0]))[img[:,0]==img.max(1)[0]]
        hue = (hue/6.0) % 1.0
        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        return hue 

    @staticmethod
    def get_v(im):
        img = im * 0.5 + 0.5
        b, c, h, w = img.shape
        it = img.transpose(1,2).transpose(2,3).contiguous().view(b, -1, c)        
        value = F.max_pool1d(it, c).view(b, h, w)
        return value 

    @staticmethod
    def get_s(im):
        img = im * 0.5 + 0.5
        b, c, h, w = img.shape
        it = img.transpose(1,2).transpose(2,3).contiguous().view(b, -1, c)        
        max_v = F.max_pool1d(it, c).view(b, h, w)
        min_v = F.max_pool1d(it*-1, c).view(b, h, w)
        satur = (max_v + min_v) / (max_v+eps)
        return satur

    def forward(self, input):
        h = self.get_h(input)
        s = self.get_s(input)
        v = self.get_v(input)
        target_h = torch.Tensor(h.shape).fill_(self.hsv[0]).to(input.device).type_as(h)
        target_s = torch.Tensor(s.shape).fill_(self.hsv[1]).to(input.device)
        target_v = torch.Tensor(v.shape).fill_(self.hsv[2]).to(input.device)
        return self.mse(h, target_h) #+ 0.4*self.mse(v, target_v)



### data loading functions
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def _rescale(img):
    return img * 2.0 - 1.0

def trans_maker(size=256):
	trans = transforms.Compose([ 
					transforms.Resize((size+36, size+36)),
					transforms.RandomHorizontalFlip(),
					transforms.RandomCrop((size, size)),
					transforms.ToTensor(),
					_rescale
					])
	return trans

def trans_maker_testing(size=256):
	trans = transforms.Compose([ 
					transforms.Resize((size, size)),
					transforms.ToTensor(),
					_rescale
					])
	return trans
transform_gan = trans_maker(size=128)

import torchvision.utils as vutils
import logging
logger = logging.getLogger(__name__)



### during training util functions
def save_image(net, dataloader_A, device, cur_iter, trial, save_path):
    """Save imag output from net"""
    logger.info('Saving gan epoch {} images: {}'.format(cur_iter, save_path))

    # Set net to evaluation mode
    net.eval()
    for p in net.parameters():
        data_type = p.type()
        break
    with torch.no_grad():
        for itx, data in enumerate(dataloader_A):
            g_img = net.gen_a2b(data[0].to(device).type(data_type))
            for i in range(g_img.size(0)):
                vutils.save_image(
                    g_img.cpu().float().add_(1).mul_(0.5),
                    os.path.join(save_path, "{}_gan_epoch_{}_iter_{}_{}.jpg".format(trial, cur_iter, itx, i)),)
    # Set net to train mode
    net.train()
    return save_path

def save_model(net, save_folder, cuda_device, if_multi_gpu, trial, cur_iter):
    """ Save current model and delete previous model, keep the saved model!"""
    save_name = "{}_gan_epoch_{}.pth".format(trial, cur_iter)
    save_path = os.path.join(save_folder, save_name)
    logger.info('Saving gan model: {}'.format(save_path))

    net.save(save_path)

    for fname in os.listdir(save_folder):
        if fname.endswith('.pth') and fname != save_name:
            delete_path = os.path.join(save_folder, fname)
            os.remove(delete_path)
            logger.info('Deleted previous gan model: {}'.format(delete_path))

    return save_path