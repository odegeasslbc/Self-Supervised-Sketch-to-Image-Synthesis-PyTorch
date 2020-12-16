import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision.datasets import ImageFolder



def make_matrix(dataset_rgb, dataset_skt, net_ae, net_ig, BATCH_SIZE, im_name):
    dataloader_rgb = iter(DataLoader(dataset_rgb, BATCH_SIZE, shuffle=True))
    dataloader_skt = iter(DataLoader(dataset_skt, BATCH_SIZE, shuffle=True))

    rgb_img = next(dataloader_rgb)
    skt_img = next(dataloader_skt)

    skt_img = skt_img.mean(dim=1, keepdim=True)
    
    image_matrix = [ torch.ones(1, 3, 512, 512) ]
    image_matrix.append(rgb_img.clone()) 
    with torch.no_grad():
        rgb_img = rgb_img.cuda()
        for skt in skt_img:
            input_skts = skt.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).cuda()

            gimg_ae, style_feats = net_ae(input_skts, rgb_img)
            g_images = net_ig(gimg_ae, style_feats).cpu()

            image_matrix.append(skt.unsqueeze(0).repeat(1,3,1,1).clone())
            image_matrix.append(gimg_ae.cpu())

            image_matrix.append(skt.unsqueeze(0).repeat(1,3,1,1).clone().fill_(1))
            image_matrix.append( torch.nn.functional.interpolate( g_images , 512 ) )

    image_matrix = torch.cat(image_matrix)
    vutils.save_image(0.5*(image_matrix+1), im_name, nrow=BATCH_SIZE+1)  

if __name__ == "__main__":
    device = 'cuda'

    from models import AE, RefineGenerator_art, RefineGenerator_face
    net_ae = AE()
    net_ae.style_encoder.reset_cls()
    net_ig = RefineGenerator_face()

    ckpt = torch.load('./models/16.pth')

    net_ae.load_state_dict(ckpt['ae'])
    net_ig.load_state_dict(ckpt['ig'])

    net_ae.to(device)
    net_ig.to(device)

    net_ae.eval()
    #net_ig.eval()

    data_root_colorful = './data/rgb/'
    #data_root_colorful = '/media/bingchen/database/images/celebaMask/CelebA_1024'
    
    data_root_sketch = './data/skt/'
    #data_root_sketch = './data/face_skt/'

    BATCH_SIZE = 3
    IM_SIZE = 512
    DATALOADER_WORKERS = 8
    
    dataset_rgb = ImageFolder(data_root_colorful, trans_maker(512))
    dataloader_rgb = iter(DataLoader(dataset_rgb, BATCH_SIZE, shuffle=True))

    dataset_skt = ImageFolder(data_root_sketch, trans_maker(512))
    dataloader_skt = iter(DataLoader(dataset_skt, BATCH_SIZE, shuffle=True))

    for i in range(10):
        make_matrix(dataloader_rgb, dataloader_skt, net_ae, net_ig, BATCH_SIZE, 'artland_matrix_%d.jpg'%i)
