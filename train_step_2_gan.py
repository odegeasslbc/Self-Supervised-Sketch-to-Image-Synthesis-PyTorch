from math import log
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import interpolate


from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
from torchvision.transforms.transforms import RandomGrayscale
from tqdm import tqdm
import pickle
import numpy as np

from datasets import PairedMultiDataset, InfiniteSamplerWrapper
from utils import copy_G_params, load_params, AverageMeter, make_folders, true_randperm, d_hinge_loss, g_hinge_loss
from models import AE, Discriminator
from generate_matrix import make_matrix

def train():
    from benchmark import calc_fid, extract_feature_from_generator_fn, load_patched_inception_v3, real_image_loader, image_generator, image_generator_perm
    import lpips

    from config import IM_SIZE_GAN, BATCH_SIZE_GAN, NFC, NBR_CLS, DATALOADER_WORKERS, EPOCH_GAN, ITERATION_AE, GAN_CKECKPOINT
    from config import SAVE_IMAGE_INTERVAL, SAVE_MODEL_INTERVAL, LOG_INTERVAL, SAVE_FOLDER, TRIAL_NAME, DATA_NAME, MULTI_GPU
    from config import FID_INTERVAL, FID_BATCH_NBR, PRETRAINED_AE_PATH
    from config import data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3
    
    real_features = None
    inception = load_patched_inception_v3().cuda()
    inception.eval()

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    saved_image_folder = saved_model_folder = None
    log_file_path = None
    if saved_image_folder is None:
        saved_image_folder, saved_model_folder = make_folders(SAVE_FOLDER, 'GAN_'+TRIAL_NAME)
        log_file_path = saved_image_folder+'/../gan_log.txt'
        log_file = open(log_file_path, 'w')
        log_file.close()

    dataset = PairedMultiDataset(data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3, im_size=IM_SIZE_GAN, rand_crop=True)
    print( 'the dataset contains %d images.'%len(dataset))
    dataloader = iter(DataLoader(dataset, BATCH_SIZE_GAN, sampler=InfiniteSamplerWrapper(dataset), num_workers=DATALOADER_WORKERS, pin_memory=True))

    from datasets import ImageFolder
    from datasets import trans_maker_augment as trans_maker

    dataset_rgb = ImageFolder(data_root_colorful, trans_maker(512))
    dataset_skt = ImageFolder(data_root_sketch_3, trans_maker(512))
    


    net_ae = AE(nfc=NFC, nbr_cls=NBR_CLS)


    if PRETRAINED_AE_PATH is None:
        PRETRAINED_AE_PATH = 'train_results/' + 'AE_' + TRIAL_NAME + '/models/%d.pth'%ITERATION_AE 
    else:
        from config import PRETRAINED_AE_ITER
        PRETRAINED_AE_PATH = PRETRAINED_AE_PATH + '/models/%d.pth'%PRETRAINED_AE_ITER 
    
    net_ae.load_state_dicts(PRETRAINED_AE_PATH)
    net_ae.cuda()
    net_ae.eval()

    RefineGenerator = None
    if DATA_NAME=='celeba':
        from models import RefineGenerator_face as RefineGenerator
    elif DATA_NAME=='art' or DATA_NAME=='shoe':
        from models import RefineGenerator_art as RefineGenerator
    net_ig = RefineGenerator(nfc=NFC, im_size=IM_SIZE_GAN).cuda()
    net_id = Discriminator(nc=3).cuda() # we use the patch_gan, so the im_size for D should be 512 even if training image size is 1024

    if MULTI_GPU:
        net_ae = nn.DataParallel(net_ae)
        net_ig = nn.DataParallel(net_ig)
        net_id = nn.DataParallel(net_id)

    net_ig_ema = copy_G_params(net_ig)

    opt_ig = optim.Adam(net_ig.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_id = optim.Adam(net_id.parameters(), lr=2e-4, betas=(0.5, 0.999))

    
    if GAN_CKECKPOINT is not None:
        ckpt = torch.load(GAN_CKECKPOINT)
        net_ig.load_state_dict(ckpt['ig'])
        net_id.load_state_dict(ckpt['id'])
        net_ig_ema = ckpt['ig_ema']
        opt_ig.load_state_dict(ckpt['opt_ig'])
        opt_id.load_state_dict(ckpt['opt_id'])

    ## create a log file
    losses_g_img = AverageMeter()
    losses_d_img = AverageMeter()
    losses_mse = AverageMeter()
    losses_rec_s = AverageMeter()

    losses_rec_ae = AverageMeter()

    fixed_skt = fixed_rgb = fixed_perm = None

    fid=[ [0,0] ]

    for epoch in range(EPOCH_GAN):   
        for iteration in tqdm(range(10000)):
            rgb_img, skt_img_1, skt_img_2, skt_img_3 = next(dataloader)
            
            rgb_img = rgb_img.cuda()

            rd = random.randint(0, 3) 
            if rd == 0:
                skt_img = skt_img_1.cuda()
            elif rd == 1:
                skt_img = skt_img_2.cuda()
            else:
                skt_img = skt_img_3.cuda()

            if iteration==0:
                fixed_skt = skt_img_3[:8].clone().cuda()
                fixed_rgb = rgb_img[:8].clone()
                fixed_perm = true_randperm(fixed_rgb.shape[0], 'cuda')

            ### 1. train D
            gimg_ae, style_feats = net_ae(skt_img, rgb_img)
            g_image = net_ig(gimg_ae, style_feats)

            pred_r = net_id( rgb_img )        
            pred_f = net_id( g_image.detach() )   

            loss_d = d_hinge_loss(pred_r, pred_f) 
            
            net_id.zero_grad()
            loss_d.backward()
            opt_id.step()
            
            loss_rec_ae = F.mse_loss(gimg_ae, rgb_img) + F.l1_loss(gimg_ae, rgb_img)
            losses_rec_ae.update(loss_rec_ae.item(), BATCH_SIZE_GAN)
                
            ### 2. train G
            pred_g = net_id(g_image)   
            loss_g = g_hinge_loss(pred_g) 

            if DATA_NAME == 'shoe':
                loss_mse = 10 * (F.l1_loss(g_image, rgb_img) + F.mse_loss(g_image, rgb_img))
            else:
                loss_mse = 10*percept( F.adaptive_avg_pool2d(g_image, output_size=256), F.adaptive_avg_pool2d(rgb_img, output_size=256)).sum()                
            losses_mse.update(loss_mse.item()/BATCH_SIZE_GAN, BATCH_SIZE_GAN)


            loss_all = loss_g + loss_mse

            if DATA_NAME == 'shoe':
                ### the grey image reconstruction
                perm = true_randperm(BATCH_SIZE_GAN)
                img_ae_perm, style_feats_perm = net_ae(skt_img, rgb_img[perm])

                gimg_grey = net_ig(img_ae_perm, style_feats_perm)
                gimg_grey = gimg_grey.mean(dim=1, keepdim=True)
                real_grey = rgb_img.mean(dim=1, keepdim=True)
                loss_rec_grey = F.mse_loss( gimg_grey , real_grey )
                loss_all += 10 * loss_rec_grey 

            net_ig.zero_grad()
            loss_all.backward()
            opt_ig.step()

            for p, avg_p in zip(net_ig.parameters(), net_ig_ema):
                avg_p.mul_(0.999).add_( p.data, alpha=0.001)

            ### 3. logging
            losses_g_img.update(pred_g.mean().item(), BATCH_SIZE_GAN)
            losses_d_img.update(pred_r.mean().item(), BATCH_SIZE_GAN)
            


            if iteration % SAVE_IMAGE_INTERVAL == 0: #show the current images
                with torch.no_grad():
                
                    backup_para_g = copy_G_params(net_ig)
                    load_params(net_ig, net_ig_ema)

                    gimg_ae, style_feats = net_ae(fixed_skt, fixed_rgb)
                    gmatch = net_ig(gimg_ae, style_feats)
                    
                    

                    gimg_ae_perm, style_feats = net_ae(fixed_skt, fixed_rgb[fixed_perm])
                    gmismatch = net_ig(gimg_ae_perm, style_feats)

                    gimg = torch.cat([ F.interpolate(fixed_rgb, IM_SIZE_GAN), 
                                       F.interpolate(fixed_skt.repeat(1,3,1,1), IM_SIZE_GAN), 
                                       gmatch, 
                                       F.interpolate(gimg_ae, IM_SIZE_GAN),
                                       gmismatch, 
                                       F.interpolate(gimg_ae_perm, IM_SIZE_GAN)])
                    
                    vutils.save_image(gimg, f'{saved_image_folder}/img_iter_{epoch}_{iteration}.jpg', normalize=True, range=(-1, 1))
                    del gimg
                    
                    make_matrix(dataset_rgb, dataset_skt, net_ae, net_ig, 5, f'{saved_image_folder}/img_iter_{epoch}_{iteration}_matrix.jpg')
                    
                    load_params(net_ig, backup_para_g)

            if iteration % LOG_INTERVAL == 0:
                log_msg = 'Iter: [{0}/{1}] G: {losses_g_img.avg:.4f}  D: {losses_d_img.avg:.4f}  MSE: {losses_mse.avg:.4f}  Rec: {losses_rec_s.avg:.5f}  FID: {fid:.4f}'.format(epoch, 
                            iteration, losses_g_img=losses_g_img, losses_d_img=losses_d_img, losses_mse=losses_mse, losses_rec_s=losses_rec_s, fid=fid[-1][0])

                print(log_msg)
                print('%.5f'%(losses_rec_ae.avg))

                if log_file_path is not None:
                    log_file = open(log_file_path, 'a')
                    log_file.write(log_msg+'\n')
                    log_file.close()
                    
                losses_g_img.reset()
                losses_d_img.reset()
                losses_mse.reset()
                losses_rec_s.reset()
                losses_rec_ae.reset()

            if iteration % SAVE_MODEL_INTERVAL ==0 or iteration+1 == 10000:
                print('Saving history model')
                torch.save( {'ig': net_ig.state_dict(),
                            'id': net_id.state_dict(),
                            'ae':net_ae.state_dict(),
                            'ig_ema': net_ig_ema,
                            'opt_ig': opt_ig.state_dict(), 
                            'opt_id': opt_id.state_dict(),
                            }, '%s/%d.pth'%(saved_model_folder, epoch))
        
            if iteration % FID_INTERVAL == 0 and iteration>1:
                print("calculating FID ...")
                fid_batch_images = FID_BATCH_NBR
                if real_features is None:
                    if os.path.exists('%s_fid_feats.npy'%(DATA_NAME)):
                        real_features = pickle.load(open('%s_fid_feats.npy'%(DATA_NAME), 'rb'))
                    else:
                        real_features = extract_feature_from_generator_fn( 
                                            real_image_loader(dataloader, n_batches=fid_batch_images), inception )
                        real_mean = np.mean(real_features, 0)
                        real_cov = np.cov(real_features, rowvar=False)
                        pickle.dump({'feats': real_features, 'mean': real_mean, 'cov': real_cov}, open('%s_fid_feats.npy'%(DATA_NAME),'wb') ) 
                        real_features = pickle.load(open('%s_fid_feats.npy'%(DATA_NAME), 'rb'))
                
                sample_features = extract_feature_from_generator_fn( image_generator(dataset, net_ae, net_ig, n_batches=fid_batch_images), inception, total=fid_batch_images )
                cur_fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])
                sample_features_perm = extract_feature_from_generator_fn( image_generator_perm(dataset, net_ae, net_ig, n_batches=fid_batch_images), inception, total=fid_batch_images )
                cur_fid_perm = calc_fid(sample_features_perm, real_mean=real_features['mean'], real_cov=real_features['cov'])
                
                fid.append( [cur_fid, cur_fid_perm] )
                print('fid:', fid)
                if log_file_path is not None:
                    log_file = open(log_file_path, 'a')
                    log_msg = 'fid: %.5f, %.5f'%(fid[-1][0], fid[-1][1])
                    log_file.write(log_msg+'\n')
                    log_file.close()

if __name__ == "__main__":
    train()