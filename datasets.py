import os
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.utils.data as data

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def _noise_adder(img):
    return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img


def _rescale(img):
    return img * 2.0 - 1.0


def trans_maker(size=512):
    trans = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    _rescale
                    ])
    return trans


def trans_maker_augment(size=256):
    trans = transforms.Compose([ 
                    transforms.Resize((int(size*1.1),int(size*1.1))),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((size, size)),
                    transforms.ToTensor(),
                    _rescale#, _noise_adder
                    ])
    return trans

from config import DATA_NAME

class SelfSupervisedDataset(Dataset):
    def __init__(self, data_root, data_root_2, im_size=512, nbr_cls=2000, rand_crop=True):
        super(SelfSupervisedDataset, self).__init__()
        self.root = data_root
        self.skt_root = data_root_2

        self.frame = self._parse_frame()
        random.shuffle(self.frame)

        self.nbr_cls = nbr_cls
        self.set_offset = 0

        self.im_size = im_size
        self.transform_rd = transforms.Compose([ 
                            transforms.Resize((int(im_size*1.3), int(im_size*1.3))),
                            transforms.RandomCrop( (int(im_size), int(im_size)) ),
                            transforms.RandomRotation( 30 ),
                            transforms.RandomHorizontalFlip(),
                            #transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            _rescale])

        self.crop = rand_crop
        if self.crop:
            self.transform_1 = transforms.Resize((int(im_size*1.1), int(im_size*1.1)))
            self.transform_2 = transforms.Compose([ transforms.ToTensor(),
                                                    _rescale
                                                    ])
            self.rand_range = int(self.im_size * 0.1)
        else:
            self.transform_normal = trans_maker(size=im_size)
    
        self.transform_flip = transforms.RandomHorizontalFlip(p=1)

        self.transform_erase = transforms.Compose([
                        transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
                        transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
                        #transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
                        #transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
                                ])

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            img_name = '%d.jpg'%(i)
            #if DATA_NAME == 'shoe':
            img_name = img_names[i]
            image_path = os.path.join(self.root, img_names[i])
            skt_path = os.path.join( self.skt_root,  img_name)
            if os.path.exists(image_path) and os.path.exists(skt_path): 
                frame.append( (image_path, skt_path) )
        return frame

    def __len__(self):
        return self.nbr_cls

    def _next_set(self):
        self.set_offset += self.nbr_cls
        if self.set_offset > ( len(self.frame) - self.nbr_cls ):
            random.shuffle(self.frame)
            self.set_offset = 0

    def __getitem__(self, idx):
        file, skt_path = self.frame[idx+self.set_offset]
        img = Image.open(file).convert('RGB')
        skt = Image.open(skt_path).convert('L')
        
        ### perform random boldness of the sketch image
        bold_factor = 3 
        skt_bold = skt.filter( ImageFilter.MinFilter(size=bold_factor) )

        if random.randint(0, 1) == 1:
            img = self.transform_flip(img)
            skt = self.transform_flip(skt)
            skt_bold = self.transform_flip(skt_bold)

        img_rd = self.transform_rd(img) 

        if self.crop:
            img_normal = self.transform_1(img) 
            skt_normal = self.transform_1(skt) 
            skt_bold = self.transform_1(skt_bold) 

            i = random.randint(0, self.rand_range) 
            j = random.randint(0, self.rand_range) 

            img_normal = F.crop(img_normal, i, j, self.im_size, self.im_size)
            skt_normal = F.crop(skt_normal, i, j, self.im_size, self.im_size)
            skt_bold = F.crop(skt_bold, i, j, self.im_size, self.im_size)

            img_normal = self.transform_2(img_normal) 
            skt_normal = self.transform_2(skt_normal) 
            skt_bold = self.transform_2(skt_bold) 
        else:
            img_normal = self.transform_normal(img)
            skt_normal = self.transform_normal(skt)
            skt_bold = self.transform_normal(skt_bold)

        skt_erased = self.transform_erase(skt_normal)
        skt_erased_bold = self.transform_erase(skt_bold)
        return img_rd, img_normal, skt_normal, skt_bold, skt_erased, skt_erased_bold, idx


class PairedMultiDataset(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, data_root_1, data_root_2, data_root_3, data_root_4, rand_crop=True, im_size=512):
        super(PairedMultiDataset, self).__init__()
        self.root_a = data_root_1
        self.root_b = data_root_2
        self.root_c = data_root_3
        self.root_d = data_root_4

        self.frame = self._parse_frame()
        
        self.crop = rand_crop
        self.im_size = im_size
        if self.crop:
            self.transform_1 = transforms.Resize((int(im_size*1.1), int(im_size*1.1)))
            self.transform_2 = transforms.Compose([ transforms.ToTensor(),
                                                    _rescale
                                                    ])
            self.rand_range = int(self.im_size * 0.1)
        else:
            self.transform = trans_maker( int( im_size ) )

    def _parse_frame(self):
        frame = []

        img_names = os.listdir(self.root_a)
        img_names.sort()
        for i in range(len(img_names)):
            img_name = '%d.jpg'%(i)
            #if DATA_NAME == 'shoe':
            img_name = img_names[i]
            image_a_path = os.path.join(self.root_a, img_names[i])
            if os.path.exists(image_a_path): 
                image_b_path = os.path.join(self.root_b, img_name)
                image_c_path = os.path.join(self.root_c, img_name)
                image_d_path = os.path.join(self.root_d, img_name)
                if os.path.exists(image_b_path) and os.path.exists(image_c_path) and os.path.exists(image_d_path):
                    frame.append( (image_a_path, image_b_path, image_c_path, image_d_path) )
                else:
                    print('2', image_a_path, image_b_path)
            else:
                print("1", image_a_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file_a, file_b, file_c, file_d = self.frame[idx]
        img_a = Image.open(file_a).convert('RGB')
        img_b = Image.open(file_b).convert('L')
        img_c = Image.open(file_c).convert('L')
        img_d = Image.open(file_d).convert('L')
            
        if self.crop:
            img_a = self.transform_1(img_a) 
            img_b = self.transform_1(img_b) 
            img_c = self.transform_1(img_c) 
            img_d = self.transform_1(img_d) 

            i = random.randint(0, self.rand_range) 
            j = random.randint(0, self.rand_range) 
            img_a = F.crop(img_a, i, j, self.im_size, self.im_size)
            img_b = F.crop(img_b, i, j, self.im_size, self.im_size)
            img_c = F.crop(img_c, i, j, self.im_size, self.im_size)
            img_d = F.crop(img_d, i, j, self.im_size, self.im_size)

            img_a = self.transform_2(img_a) 
            img_b = self.transform_2(img_b) 
            img_c = self.transform_2(img_c) 
            img_d = self.transform_2(img_d) 
        else:
            img_a = self.transform(img_a) 
            img_b = self.transform(img_b) 
            img_c = self.transform(img_c) 
            img_d = self.transform(img_d) 


        return (img_a, img_b, img_c, img_d)


class PairedDataset(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, data_root_1, data_root_2, transform=trans_maker(512)):
        super(PairedDataset, self).__init__()
        self.root_a = data_root_1
        self.root_b = data_root_2

        self.frame = self._parse_frame()
        self.transform = transform


    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root_a)
        img_names.sort()
        for i in range(len(img_names)):
            img_name = '%d.jpg'%(i)
            if DATA_NAME == 'shoe':
                img_name = img_names[i]
            image_a_path = os.path.join(self.root_a, img_names[i])
            if ('.jpg' in image_a_path) or ('.png' in image_a_path): 
                image_b_path = os.path.join(self.root_b, img_name)
                if os.path.exists(image_b_path):
                    frame.append( (image_a_path, image_b_path) )

        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file_a, file_b = self.frame[idx]
        img_a = Image.open(file_a).convert('RGB')
        img_b = Image.open(file_b).convert('L')
            
        if self.transform:
            img_a = self.transform(img_a) 
            img_b = self.transform(img_b) 

        return (img_a, img_b)


class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, data_root, transform=trans_maker(512)):
        super( ImageFolder, self).__init__()
        self.root = data_root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if ('.jpg' in image_path) or ('.png' in image_path): 
                frame.append(image_path)

        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 
        return img




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