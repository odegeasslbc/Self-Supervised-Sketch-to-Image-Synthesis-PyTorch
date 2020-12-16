import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import random
from tqdm import tqdm

from datasets import SelfSupervisedDataset, PairedMultiDataset, InfiniteSamplerWrapper
from utils import make_folders, AverageMeter, true_randperm
from models import StyleEncoder, ContentEncoder, Decoder


def loss_for_list(loss, fl1, fl2, detach_second=True):
    result_loss = 0
    for f_idx in range(len(fl1)):
        if detach_second:
            result_loss += loss( fl1[f_idx] , fl2[f_idx].detach() )
        else:
            result_loss += loss( fl1[f_idx] , fl2[f_idx] )
    return result_loss


def loss_for_list_perm(loss, fl1, fl2, detach_second=True):
    result_loss = 0
    for f_idx in range(len(fl1)):
        perm = true_randperm(fl1[0].shape[0], fl1[0].device)
        if detach_second:
            result_loss += F.relu( 2 + loss( fl1[f_idx] , fl2[f_idx].detach() ) - loss( fl1[f_idx][perm] , fl2[f_idx].detach() ))
        else:
            result_loss += F.relu( 2 + loss( fl1[f_idx] , fl2[f_idx] ) - loss( fl1[f_idx][perm] , fl2[f_idx] ))
    return result_loss


def loss_for_list_mean(feat_list):
    loss = 0
    for feat in feat_list:
        if len(feat.shape) == 4:
            feat = feat.mean(dim=[2,3])
            loss += F.l1_loss( feat, torch.ones_like(feat) )
        else:
            loss += F.l1_loss( feat, torch.zeros_like(feat) )
    return loss


def train():
    from config import IM_SIZE_AE, BATCH_SIZE_AE, NFC, NBR_CLS, DATALOADER_WORKERS, ITERATION_AE
    from config import SAVE_IMAGE_INTERVAL, SAVE_MODEL_INTERVAL, SAVE_FOLDER, TRIAL_NAME, LOG_INTERVAL
    from config import DATA_NAME
    from config import data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3
    

    dataset = PairedMultiDataset(data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3, im_size=IM_SIZE_AE, rand_crop=True)
    print(len(dataset))
    dataloader = iter(DataLoader(dataset, BATCH_SIZE_AE, \
        sampler=InfiniteSamplerWrapper(dataset), num_workers=DATALOADER_WORKERS, pin_memory=True))


    dataset_ss = SelfSupervisedDataset(data_root_colorful, data_root_sketch_3, im_size=IM_SIZE_AE, nbr_cls=NBR_CLS, rand_crop=True)
    print(len(dataset_ss), len(dataset_ss.frame))
    dataloader_ss = iter(DataLoader(dataset_ss, BATCH_SIZE_AE, \
        sampler=InfiniteSamplerWrapper(dataset_ss), num_workers=DATALOADER_WORKERS, pin_memory=True))


    style_encoder = StyleEncoder(nfc=NFC, nbr_cls=NBR_CLS).cuda()
    content_encoder = ContentEncoder(nfc=NFC).cuda()
    decoder = Decoder(nfc=NFC).cuda()

    opt_c = optim.Adam(content_encoder.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_s = optim.Adam( style_encoder.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(decoder.parameters(), lr=2e-4, betas=(0.5, 0.999))

    style_encoder.reset_cls()
    style_encoder.final_cls.cuda()

    from config import PRETRAINED_AE_PATH, PRETRAINED_AE_ITER
    if PRETRAINED_AE_PATH is not None:
        PRETRAINED_AE_PATH = PRETRAINED_AE_PATH + '/models/%d.pth'%PRETRAINED_AE_ITER 
        ckpt = torch.load(PRETRAINED_AE_PATH)
        
        print(PRETRAINED_AE_PATH)
        
        style_encoder.load_state_dict(ckpt['s'])
        content_encoder.load_state_dict(ckpt['c'])
        decoder.load_state_dict(ckpt['d'])

        opt_c.load_state_dict(ckpt['opt_c'])
        opt_s.load_state_dict(ckpt['opt_s'])
        opt_d.load_state_dict(ckpt['opt_d'])
        print('loaded pre-trained AE')
    
    style_encoder.reset_cls()
    style_encoder.final_cls.cuda()
    opt_s_cls = optim.Adam( style_encoder.final_cls.parameters(), lr=2e-4, betas=(0.5, 0.999))


    saved_image_folder, saved_model_folder = make_folders(SAVE_FOLDER, 'AE_'+TRIAL_NAME)
    log_file_path = saved_image_folder+'/../ae_log.txt'
    log_file = open(log_file_path, 'w')
    log_file.close()
    ## for logging
    losses_sf_consist = AverageMeter()
    losses_cf_consist = AverageMeter()
    losses_cls = AverageMeter()
    losses_rec_rd = AverageMeter()
    losses_rec_org = AverageMeter()
    losses_rec_grey = AverageMeter()

    
    import lpips
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    for iteration in tqdm(range(ITERATION_AE)):
        
        if iteration%( (NBR_CLS*100)//BATCH_SIZE_AE )==0 and iteration>1:
            dataset_ss._next_set()
            dataloader_ss = iter(DataLoader(dataset_ss, BATCH_SIZE_AE, sampler=InfiniteSamplerWrapper(dataset_ss), num_workers=DATALOADER_WORKERS, pin_memory=True))
            style_encoder.reset_cls()
            opt_s_cls = optim.Adam( style_encoder.final_cls.parameters(), lr=2e-4, betas=(0.5, 0.999))
            
            opt_s.param_groups[0]['lr'] = 1e-4
            opt_d.param_groups[0]['lr'] = 1e-4

        ### 1. train the encoder with self-supervision methods
        rgb_img_rd, rgb_img_org, skt_org, skt_bold, skt_erased, skt_erased_bold, img_idx = next(dataloader_ss)
        rgb_img_rd = rgb_img_rd.cuda()
        rgb_img_org = rgb_img_org.cuda()
        img_idx = img_idx.cuda()

        skt_org = F.interpolate( skt_org , size=512 ).cuda()
        skt_bold = F.interpolate( skt_bold , size=512 ).cuda()
        skt_erased = F.interpolate( skt_erased , size=512 ).cuda()
        skt_erased_bold = F.interpolate( skt_erased_bold , size=512 ).cuda()

        style_encoder.zero_grad()
        decoder.zero_grad()
        content_encoder.zero_grad()

        style_vector_rd, pred_cls_rd = style_encoder(rgb_img_rd)
        style_vector_org, pred_cls_org = style_encoder(rgb_img_org)
        
        content_feats = content_encoder(skt_org)
        content_feats_bold = content_encoder(skt_bold)
        content_feats_erased = content_encoder(skt_erased)
        content_feats_eb = content_encoder(skt_erased_bold)
        
        rd = random.randint(0, 3)
        gimg_rd = None
        if rd==0:
            gimg_rd = decoder(content_feats, style_vector_rd)
        elif rd==1:
            gimg_rd = decoder(content_feats_bold, style_vector_rd)
        elif rd==2:
            gimg_rd = decoder(content_feats_erased, style_vector_rd)
        elif rd==3:
            gimg_rd = decoder(content_feats_eb, style_vector_rd)


        loss_cf_consist = loss_for_list_perm(F.mse_loss, content_feats_bold, content_feats) +\
                            loss_for_list_perm(F.mse_loss, content_feats_erased, content_feats) +\
                                loss_for_list_perm(F.mse_loss, content_feats_eb, content_feats)

        loss_sf_consist = 0
        for loss_idx in range(3):
            loss_sf_consist += -F.cosine_similarity(style_vector_rd[loss_idx], style_vector_org[loss_idx].detach()).mean() + \
                                    F.cosine_similarity(style_vector_rd[loss_idx], style_vector_org[loss_idx][torch.randperm(BATCH_SIZE_AE)].detach()).mean()
        
        loss_cls = F.cross_entropy(pred_cls_rd, img_idx) + F.cross_entropy(pred_cls_org, img_idx)
        loss_rec_rd = F.mse_loss(gimg_rd, rgb_img_org)
        if DATA_NAME != 'shoe':
            loss_rec_rd += percept( F.adaptive_avg_pool2d(gimg_rd, output_size=256), F.adaptive_avg_pool2d(rgb_img_org, output_size=256)).sum()                
        else:
            loss_rec_rd += F.l1_loss(gimg_rd, rgb_img_org)
        
        loss_total = loss_cls + loss_sf_consist + loss_rec_rd + loss_cf_consist #+ loss_kl_c + loss_kl_s
        loss_total.backward()

        opt_s.step()
        opt_s_cls.step()
        opt_c.step()
        opt_d.step()
        
        ### 2. train as AutoEncoder
        rgb_img, skt_img_1, skt_img_2, skt_img_3 = next(dataloader)
            
        rgb_img = rgb_img.cuda()

        rd = random.randint(0, 3) 
        if rd == 0:
            skt_img = skt_img_1
        elif rd == 1:
            skt_img = skt_img_2
        else:
            skt_img = skt_img_3

        skt_img = F.interpolate(skt_img, size=512).cuda()

        style_encoder.zero_grad()
        decoder.zero_grad()
        content_encoder.zero_grad()

        style_vector, _ = style_encoder(rgb_img)
        content_feats = content_encoder(skt_img)
        gimg = decoder(content_feats, style_vector)

        loss_rec_org = F.mse_loss(gimg, rgb_img)
        if DATA_NAME != 'shoe':
            loss_rec_org += percept( F.adaptive_avg_pool2d(gimg, output_size=256), 
                                F.adaptive_avg_pool2d(rgb_img, output_size=256)).sum()                
        #else:
        #    loss_rec_org += F.l1_loss(gimg, rgb_img)
            
        loss_rec = loss_rec_org 
        if DATA_NAME == 'shoe':
            ### the grey image reconstruction
            perm = true_randperm(BATCH_SIZE_AE)
            gimg_perm = decoder(content_feats, [s[perm] for s in style_vector])
            gimg_grey = gimg_perm.mean(dim=1, keepdim=True)
            real_grey = rgb_img.mean(dim=1, keepdim=True)
            loss_rec_grey = F.mse_loss( gimg_grey , real_grey )
            loss_rec += loss_rec_grey 
        loss_rec.backward()

        opt_s.step()
        opt_d.step()
        opt_c.step()

        ### Logging
        losses_cf_consist.update(loss_cf_consist.mean().item(), BATCH_SIZE_AE)
        losses_sf_consist.update(loss_sf_consist.mean().item(), BATCH_SIZE_AE)
        losses_cls.update(loss_cls.mean().item(), BATCH_SIZE_AE)
        losses_rec_rd.update(loss_rec_rd.item(), BATCH_SIZE_AE)
        losses_rec_org.update(loss_rec_org.item(), BATCH_SIZE_AE)
        if DATA_NAME=='shoe':
            losses_rec_grey.update(loss_rec_grey.item(), BATCH_SIZE_AE)


        if iteration%LOG_INTERVAL==0:
            log_msg = 'Train Stage 1: AE: \nrec_rd: %.4f  rec_org: %.4f  cls: %.4f  style_consist: %.4f  content_consist: %.4f  rec_grey: %.4f'%(losses_rec_rd.avg, \
                    losses_rec_org.avg, losses_cls.avg, losses_sf_consist.avg, losses_cf_consist.avg, losses_rec_grey.avg)
            
            print(log_msg)

            if log_file_path is not None:
                log_file = open(log_file_path, 'a')
                log_file.write(log_msg+'\n')
                log_file.close()

            losses_sf_consist.reset()
            losses_cls.reset()
            losses_rec_rd.reset()
            losses_rec_org.reset()
            losses_cf_consist.reset()
            losses_rec_grey.reset()

        if iteration%SAVE_IMAGE_INTERVAL==0:
            vutils.save_image( torch.cat([rgb_img_rd, F.interpolate(skt_org.repeat(1,3,1,1), size=512) , gimg_rd]), '%s/rd_%d.jpg'%(saved_image_folder, iteration), normalize=True, range=(-1,1) )
            if DATA_NAME != 'shoe':
                with torch.no_grad():
                    perm = true_randperm(BATCH_SIZE_AE)
                    gimg_perm = decoder([c for c in content_feats], [s[perm] for s in style_vector])
            vutils.save_image( torch.cat([rgb_img, F.interpolate(skt_img.repeat(1,3,1,1), size=512), gimg, gimg_perm]), '%s/org_%d.jpg'%(saved_image_folder, iteration), normalize=True, range=(-1,1) )

        if iteration%SAVE_MODEL_INTERVAL==0:
            print('Saving history model')
            torch.save( {'s': style_encoder.state_dict(),
                        'd': decoder.state_dict(),
                        'c': content_encoder.state_dict(),
                        'opt_c': opt_c.state_dict(),
                        'opt_s_cls': opt_s_cls.state_dict(),
                        'opt_s': opt_s.state_dict(), 
                        'opt_d': opt_d.state_dict(),
                            }, '%s/%d.pth'%(saved_model_folder, iteration))
    
    torch.save( {'s': style_encoder.state_dict(),
                        'd': decoder.state_dict(),
                        'c': content_encoder.state_dict(),
                        'opt_c': opt_c.state_dict(),
                        'opt_s_cls': opt_s_cls.state_dict(),
                        'opt_s': opt_s.state_dict(), 
                        'opt_d': opt_d.state_dict(),
                            }, '%s/%d.pth'%(saved_model_folder, ITERATION_AE))


if __name__ == "__main__":
    train()