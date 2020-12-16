import torch
from torch.utils.data import DataLoader
from torch.utils.data import dataloader
from torchvision import utils as vutils
from torchvision.datasets import ImageFolder

from refine_ae_as_gan import RefineGenerator, AE
from utils import InfiniteSamplerWrapper, PairedMultiDataset, true_randperm, trans_maker



def make_matrix(dataloader_rgb, dataloader_skt, net_ae, net_ig, BATCH_SIZE, im_name):
    rgb_img = next(dataloader_rgb)[0]
    skt_img = next(dataloader_skt)[0]
    skt_img = skt_img.mean(dim=1, keepdim=True)
    
    new_rgb_1 = torch.cat([torch.cat([rgb_img[0], rgb_img[1]], dim=1), 
                torch.cat([rgb_img[2], rgb_img[3]], dim=1)], dim=1) 
    new_rgb_1 = torch.nn.functional.interpolate(new_rgb_1.unsqueeze(0), size = 512)
    
    new_rgb_2 = torch.cat([torch.cat([rgb_img[0], rgb_img[1]], dim=2), 
                torch.cat([rgb_img[2], rgb_img[3]], dim=2)], dim=2) 
    new_rgb_2 = torch.nn.functional.interpolate(new_rgb_2.unsqueeze(0), size = 512)

    new_rgb_3 = torch.cat([torch.cat([rgb_img[0], rgb_img[1]], dim=1), 
                torch.cat([rgb_img[2], rgb_img[3]], dim=1)], dim=2) 
    new_rgb_3 = torch.nn.functional.interpolate(new_rgb_3.unsqueeze(0), size = 512)

    new_rgb_4 = torch.cat([torch.cat([rgb_img[0], rgb_img[1]], dim=2), 
                torch.cat([rgb_img[2], rgb_img[3]], dim=2)], dim=1) 
    new_rgb_4 = torch.nn.functional.interpolate(new_rgb_4.unsqueeze(0), size = 512)


    rgb_img = torch.cat( [new_rgb_1, new_rgb_2, new_rgb_3, new_rgb_4] )

    image_matrix = [ torch.ones(1, 3, 512, 512) ]
    image_matrix.append(rgb_img.clone()) 
    with torch.no_grad():
        rgb_img = rgb_img.cuda()
        for skt in skt_img:
            input_skts = skt.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).cuda()

            gimg_ae, style_feats = net_ae(input_skts, rgb_img)
            g_images = net_ig(gimg_ae, style_feats).cpu()

            #image_matrix.append(skt.unsqueeze(0).repeat(1,3,1,1).clone())
            #image_matrix.append(gimg_ae.cpu())

            image_matrix.append(skt.unsqueeze(0).repeat(1,3,1,1).clone())
            image_matrix.append(g_images)

    image_matrix = torch.cat(image_matrix)
    vutils.save_image(0.5*(image_matrix+1), im_name, nrow=BATCH_SIZE+1)  

if __name__ == "__main__":
    device = 'cuda'

    net_ae = AE()
    net_ae.style_encoder.reset_cls()
    net_ig = RefineGenerator()

    net_ae = torch.nn.DataParallel(net_ae)
    net_ig = torch.nn.DataParallel(net_ig)

    ckpt = torch.load('./models/15.pth')

    net_ae.load_state_dict(ckpt['ae'])
    net_ig.load_state_dict(ckpt['ig'])
    from utils import load_params
    load_params(net_ig, ckpt['ig_ema'])

    net_ae.to(device)
    net_ig.to(device)

    net_ae.eval()
    #net_ig.eval()

    data_root_colorful = '../artland_1/data/rgb/'
    #data_root_colorful = '/media/bingchen/database/images/celebaMask/CelebA_1024'
    
    data_root_sketch = '../artland_1/data/skt/'
    #data_root_sketch = './data/face_skt/'

    BATCH_SIZE = 4
    IM_SIZE = 512
    DATALOADER_WORKERS = 8
    
    dataset_rgb = ImageFolder(data_root_colorful, trans_maker(512))
    dataloader_rgb = iter(DataLoader(dataset_rgb, BATCH_SIZE, shuffle=True))

    dataset_skt = ImageFolder(data_root_sketch, trans_maker(512))
    dataloader_skt = iter(DataLoader(dataset_skt, BATCH_SIZE, shuffle=True))

    for i in range(10):
        make_matrix(dataloader_rgb, dataloader_skt, net_ae, net_ig, BATCH_SIZE, 'artland_mix_%d.jpg'%i)
