import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import dataloader
from torchvision import utils as vutils
import torchvision
from torchvision.datasets import ImageFolder

from refine_ae_as_gan import RefineGenerator, AE
from utils import InfiniteSamplerWrapper, PairedMultiDataset, true_randperm, trans_maker


from sketch_models import Generator as ToSktGenerator
from sketch_models import VGGSimple

device = 'cuda'

vgg = VGGSimple()
vgg.load_state_dict(torch.load('../../sketch_styletransfer/vgg-feature-weights.pth', map_location=lambda a,b:a))
vgg.to(device)
vgg.eval()
for p in vgg.parameters():
    p.requires_grad = False

    
net_2skt = ToSktGenerator(infc=256, nfc=128)
checkpoint = torch.load('../../sketch_styletransfer/train_results/unsplash/models/2799_model.pth', map_location=lambda storage, loc: storage)
net_2skt.load_state_dict(checkpoint['g'])
print("To-Skt model loaded")

net_2skt.to(device)   
net_2skt.eval()

net_ae = AE()
net_ae.style_encoder.reset_cls()
net_ig = RefineGenerator()

ckpt = torch.load('./models/15.pth')

net_ae = torch.nn.DataParallel(net_ae)
net_ig = torch.nn.DataParallel(net_ig)

net_ae.load_state_dict(ckpt['ae'])
net_ig.load_state_dict(ckpt['ig'])

net_ae.to(device)
net_ig.to(device)

net_ae.eval()

batch_size = 8
dataset = ImageFolder(root='../artland_1/data/rgb_select/', transform=trans_maker(size=512)) 
dataset_pr = ImageFolder(root='../../data/unsplash/', transform=trans_maker(size=512))

dataloader = iter(DataLoader(dataset, batch_size, \
        sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=True))

dataloader_pr = iter(DataLoader(dataset_pr, batch_size, \
        sampler=InfiniteSamplerWrapper(dataset_pr), num_workers=4, pin_memory=True))


for k in range(20):
    rgb_images = next(dataloader_pr)[0].to(device)

    skt_org_imgs = rgb_images[batch_size//2:].clone()

    skt_imgs = net_2skt(vgg( F.interpolate( skt_org_imgs , 256 ) , base=8 )[2])
    skt_imgs = skt_imgs.mean(1, keepdim=True)
    skt_imgs = (skt_imgs > 0.7).float()
    #skt_imgs = skt_imgs*2 - 1
    skt_imgs = F.interpolate(skt_imgs, size=512)

    vutils.save_image(skt_imgs.add(1).mul(0.5), 'tmp_skt.jpg')

    #sty_imgs = rgb_images[batch_size//2:]
    sty_imgs = next(dataloader)[0][batch_size//2:].to(device)
    img_to_save = [torch.ones(1,3,512,512)]
    img_to_save.append(sty_imgs.cpu())
    with torch.no_grad():
        j = 0
        for skt in skt_imgs:
            img_to_save.append( skt_org_imgs[j].view(1,3,512,512).cpu() )
            j += 1
            skt = skt.unsqueeze(0).repeat(batch_size//2,1,1,1)
            skt = skt.mean(dim=1, keepdim=True)
            g_images = net_ig(*net_ae(skt, sty_imgs)).cpu()

            img_to_save.append(g_images)

    img_to_save = torch.cat(img_to_save)
    vutils.save_image(img_to_save.add(1).mul(0.5), 'style_transfer_prart_%d.jpg'%(k+10), nrow=batch_size//2+1)