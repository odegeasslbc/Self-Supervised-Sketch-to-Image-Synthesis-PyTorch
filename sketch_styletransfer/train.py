import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader

from models import Generator, Discriminator, VGGSimple, Adaptive_pool, adain, get_batched_gram_matrix
from operation import InfiniteSamplerWrapper, trans_maker

import argparse
import tqdm


torch.backends.cudnn.benchmark = True



def creat_folder(save_folder, trial_name):
    saved_model_folder = os.path.join(save_folder, 'train_results/%s/models'%trial_name)
    saved_image_folder = os.path.join(save_folder, 'train_results/%s/images'%trial_name)
    folders = [os.path.join(save_folder, 'train_results'), os.path.join(save_folder, 'train_results/%s'%trial_name), 
    os.path.join(save_folder, 'train_results/%s/images'%trial_name), os.path.join(save_folder, 'train_results/%s/models'%trial_name)]

    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    return saved_model_folder, saved_image_folder

def train_d(net, data, label="real"):
    pred = net(data)
    if label=="real":
        err = F.relu(1-pred).mean()
    else:
        err = F.relu(1+pred).mean()

    err.backward()
    return torch.sigmoid(pred).mean().item()

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def gram_loss(input, target):
    in_gram = gram_matrix(input)
    tar_gram = gram_matrix(target.detach())
    return F.mse_loss(in_gram, tar_gram)

def save_image(net_g, dataloader, saved_image_folder, n_iter):
    net_g.eval()
    with torch.no_grad():
        imgs = []
        real = []
        for i, d in enumerate(dataloader):
            if i < 2:
                f_3 = vgg(d[0].to(device), base=base)[2]
                imgs.append(net_g(f_3).cpu())
                real.append(d[0])
            else:
                break
        imgs = torch.cat(imgs, dim=0)
        real = torch.cat(real, dim=0)
        sss = torch.cat([imgs, real], dim=0)
        
        vutils.save_image( sss, "%s/iter_%d.jpg"%(saved_image_folder, n_iter), range=(-1,1), normalize=True)
        del imgs
    net_g.train()

def train(net_g, net_d_style, max_iteration):
    print('training begin ... ')
    titles = ['D_r', 'D_f', 'G', 'G_rec']
    losses = {title: 0.0 for title in titles}

    saved_model_folder, saved_image_folder = creat_folder(save_folder, trial_name)
    
    for n_iter in tqdm.tqdm(range(max_iteration+1)):
        if (n_iter+1)%(100)==0:
            try:
                model_dict = {'g': net_g.state_dict(), 'ds':net_d_style.state_dict()}
                torch.save(model_dict, os.path.join(saved_model_folder, '%d_model.pth'%(n_iter)))
                opt_dict = {'g': optG.state_dict(), 'ds':optDS.state_dict()}
                torch.save(opt_dict, os.path.join(saved_model_folder, '%d_opt.pth'%(n_iter)))
            except:
                print("models not properly saved")
        if n_iter%100==0:
            save_image(net_g, dataloader_A_fixed, saved_image_folder, n_iter)
        
        ## 1. prepare data
        real_style = next(dataloader_B)[0].to(device)
        real_content = next(dataloader_A)[0].to(device)
     
        cf_1, cf_2, cf_3, cf_4, cf_5 = vgg(real_content, base=base)
        sf_1, sf_2, sf_3, sf_4, sf_5 = vgg(real_style, base=base)

        fake_img = net_g(cf_3)
        tf_1, tf_2, tf_3, tf_4, tf_5 = vgg(fake_img, base=base)

        target_3 = adain(cf_3, sf_3)

        gram_sf_4 = gram_reshape(get_batched_gram_matrix(sf_4))
        gram_sf_3 = gram_reshape(get_batched_gram_matrix(sf_3))
        gram_sf_2 = gram_reshape(get_batched_gram_matrix(sf_2))
        real_style_sample = torch.cat([gram_sf_2, gram_sf_3, gram_sf_4], dim=1)

        gram_tf_4 = gram_reshape(get_batched_gram_matrix(tf_4))
        gram_tf_3 = gram_reshape(get_batched_gram_matrix(tf_3))
        gram_tf_2 = gram_reshape(get_batched_gram_matrix(tf_2))
        fake_style_sample = torch.cat([gram_tf_2, gram_tf_3, gram_tf_4], dim=1)

        ## 3. train Discriminator
        net_d_style.zero_grad()
        
        ### 3.1. train D_style on real data
        D_R = train_d(net_d_style, real_style_sample, label="real")
        ### 3.2. train D_style on fake data
        D_F = train_d(net_d_style, fake_style_sample.detach(), label="fake")
        optDS.step()
        
        ## 2. train Generator
        net_g.zero_grad()
        ### 2.1. train G as real image
        pred_gs = net_d_style(fake_style_sample)
        err_gs = -pred_gs.mean()
   
        G_B = torch.sigmoid(pred_gs).mean().item() #+ torch.sigmoid(pred_gc).mean().item()

        err_rec = F.mse_loss(tf_3, target_3)
        err_gram = 2000*(
            gram_loss(tf_4, sf_4) + \
                gram_loss(tf_3, sf_3) + \
                    gram_loss(tf_2, sf_2))

        G_rec = err_gram.item()

        err = err_gs + mse_weight*err_rec + gram_weight*err_gram
        err.backward()

        optG.step()

        ## logging ~
        loss_values = [D_R, D_F, G_B, G_rec]
        for i, term in enumerate(titles):
            losses[term] += loss_values[i]
        
        if n_iter > 0 and n_iter % log_interval == 0:
            log_line = ""
            for key, value in losses.items():
                log_line += "%s: %.5f  "%(key, value/log_interval)
                losses[key] = 0
            print(log_line)



if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='Style transfer GAN, during training, the model will learn to take a image from one specific catagory and transform it into another style domain')

    parser.add_argument('--path_a', type=str, help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--path_b', type=str, default='./sketches/', help='path of target dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default is 2e-4, usually dont need to change it, you can try make it smaller, such as 1e-4')
    parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
    parser.add_argument('--total_iter', type=int, default=5000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--mse_weight', default=0.2, type=float, help='let G generate images with content more like in set A')
    parser.add_argument('--gram_weight', default=1, type=float, help='let G generate images with style more like in set B')
    parser.add_argument('--checkpoint', default='None', type=str, help='specify the path of the pre-trained model')
    
    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    data_root_A = args.path_a
    data_root_B = args.path_b
    mse_weight = args.mse_weight
    gram_weight = args.gram_weight
    max_iteration = args.total_iter
    device = torch.device("cuda:%d"%(args.gpu_id))

    im_size = args.im_size
    if im_size == 128:
        base = 4
    elif im_size == 256:
        base = 8
    elif im_size == 512:
        base = 16
    if im_size not in [128, 256, 512]:
        print("the size must be in [128, 256, 512]")
  

    log_interval = 100
    save_folder = './'
    number_model_to_save = 30

    vgg = VGGSimple()
    vgg.load_state_dict(torch.load('./vgg-feature-weights.pth', map_location=lambda a,b:a))
    vgg.to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    dataset_A = Dataset.ImageFolder(root=data_root_A, transform=trans_maker(args.im_size)) 
    dataloader_A_fixed = DataLoader(dataset_A, 8, shuffle=False, num_workers=4)
    dataloader_A = iter(DataLoader(dataset_A, args.batch_size, shuffle=False,\
            sampler=InfiniteSamplerWrapper(dataset_A), num_workers=4, pin_memory=True))

    dataset_B = Dataset.ImageFolder(root=data_root_B, transform=trans_maker(args.im_size)) 
    dataloader_B = iter(DataLoader(dataset_B, args.batch_size, shuffle=False,\
            sampler=InfiniteSamplerWrapper(dataset_B), num_workers=4, pin_memory=True))

    net_g = Generator(infc=256, nfc=128)

    net_d_style = Discriminator(nfc=128*3, norm_layer=nn.BatchNorm2d)
    gram_reshape = Adaptive_pool(128, 16)
    # this style discriminator take input: 512x512 gram matrix from 512x8x8 vgg feature,
    # the reshaped pooled input size is: 256x16x16
    
    if args.checkpoint is not 'None':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        net_g.load_state_dict(checkpoint['g'])
        net_d_style.load_state_dict(checkpoint['ds'])
        print("saved model loaded")

    net_d_style.to(device)
    net_g.to(device)   

    optG = optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.99))
    optDS = optim.Adam(net_d_style.parameters(), lr=args.lr, betas=(0.5, 0.99))
    
    if args.checkpoint is not 'None':
        opt_path = args.checkpoint.replace("_model.pth", "_opt.pth")
        try:
            opt_weights = torch.load(opt_path, map_location=lambda a, b: a)
            optG.load_state_dict(opt_weights['g'])
            optDS.load_state_dict(opt_weights['ds'])
            print("saved optimizer loaded")
        except:
            print("no optimizer weights detected, resuming a training without optimizer weights may not let the model converge as desired")
            pass

    
    train(net_g, net_d_style, max_iteration)
