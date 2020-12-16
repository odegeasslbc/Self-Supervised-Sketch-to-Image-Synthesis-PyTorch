import os

import torch
import torchvision.datasets as Dataset
import torchvision.utils as vutils
from torch import nn


from models import Generator, VGGSimple
from operation import trans_maker_testing

import argparse


if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='Style transfer GAN, during training, the model will learn to take a image from one specific catagory and transform it into another style domain')

    parser.add_argument('--path_content', type=str, help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--path_result', type=str, help='path to save the result images')
    parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--norm_layer', type=str, default="instance", help='can choose between [batch, instance]')
    parser.add_argument('--checkpoint', type=str, help='specify the path of the pre-trained model')
    
    args = parser.parse_args()

    print(str(args))

    device = torch.device("cuda:%d"%(args.gpu_id))

    im_size = args.im_size
    if im_size == 128:
        base = 4
    elif im_size == 256:
        base = 8
    elif im_size == 512:
        base = 16
    elif im_size == 1024:
        base = 32
    if im_size not in [128, 256, 512, 1024]:
        print("the size must be in [128, 256, 512, 1024]")
  
    vgg = VGGSimple()
    vgg.load_state_dict(torch.load('./vgg-feature-weights.pth', map_location=lambda a,b:a))
    vgg.to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    dataset = Dataset.ImageFolder(root=args.path_content, transform=trans_maker_testing(size=args.im_size)) 
    
    net_g = Generator(infc=256, nfc=128)
    
    if args.checkpoint is not 'None':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        net_g.load_state_dict(checkpoint['g'])
        print("saved model loaded")

    net_g.to(device)   
    net_g.eval()

    dist_path = args.path_result
    if not os.path.exists(dist_path):
        os.mkdir(dist_path)


    print("begin generating images ...")
    with torch.no_grad():
        for i in range(len(dataset)):
            print("generating the %dth image"%(i))
            img = dataset[i][0].to(device)
            feat = vgg(img, base=base)[2]
            g_img = net_g(feat)

            g_img = g_img.mean(1).unsqueeze(1).detach().add(1).mul(0.5)
            g_img = (g_img > 0.7).float()
            vutils.save_image(g_img, os.path.join(dist_path, '%d.jpg'%(i)))