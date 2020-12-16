from config import DATA_NAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_dwconv import DepthwiseConv2d

import math
import random


def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    except:
        pass


class DMI(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.weight_a = nn.Parameter(torch.ones(1, in_channels, 1, 1)*1.01)
        self.weight_b = nn.Parameter(torch.ones(1, in_channels, 1, 1)*0.99)

        self.bias_a = nn.Parameter(torch.zeros(1, in_channels, 1, 1)+0.01)
        self.bias_b = nn.Parameter(torch.zeros(1, in_channels, 1, 1)-0.01)

    def forward(self, feat, mask):
        if feat.shape[1] > mask.shape[1]:
            channel_scale = feat.shape[1] // mask.shape[1]
            mask = mask.repeat(1, channel_scale, 1, 1)
        
        mask = F.interpolate(mask, size=feat.shape[2])
        feat_a = self.weight_a * feat * mask + self.bias_a
        feat_b = self.weight_b * feat * (1-mask) + self.bias_b
        return feat_a + feat_b


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class Squeeze(nn.Module):
    def forward(self, feat):
        return feat.squeeze(-1).squeeze(-1)


class UnSqueeze(nn.Module):
    def forward(self, feat):
        return feat.unsqueeze(-1).unsqueeze(-1)


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, ch, 1, 1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            spectral_norm( nn.Linear(channel, channel // reduction, bias=False) ),
            nn.ReLU(inplace=True),
            spectral_norm( nn.Linear(channel // reduction, channel, bias=False) ),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlkG(nn.Module):
    def __init__(self, ch, ch_m=4):
        super().__init__()
        self.main = nn.Sequential(spectral_norm( nn.BatchNorm2d(ch) ),
                            spectral_norm( nn.Conv2d(ch, ch*ch_m, 1, 1, 0, bias=False) ),
                            spectral_norm( nn.BatchNorm2d(ch*ch_m) ), Swish(),
                            spectral_norm( DepthwiseConv2d(ch*ch_m, ch*ch_m, 5, 1, 2) ),
                            spectral_norm( nn.BatchNorm2d(ch*ch_m) ), Swish(),
                            spectral_norm( nn.Conv2d(ch*ch_m, ch, 1, 1, 0, bias=False) ),
                            spectral_norm( nn.BatchNorm2d(ch) ),
                            SELayer(ch))
    def forward(self, feat):
        return feat + self.main(feat)


class ResBlkE(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.main = nn.Sequential(
                            spectral_norm( nn.BatchNorm2d(ch) ), Swish(),
                            spectral_norm( nn.Conv2d(ch, ch, 3, 1, 1, bias=False) ),
                            spectral_norm( nn.BatchNorm2d(ch) ), Swish(),
                            spectral_norm( nn.Conv2d(ch, ch, 3, 1, 1, bias=False) ),
                            SELayer(ch))

    def forward(self, feat):
        return feat + self.main(feat)


class StyleEncoder(nn.Module):
    def __init__(self, nfc=64, nbr_cls=500):
        super().__init__()

        self.nfc = nfc

        self.sf_256 = nn.Sequential(nn.Conv2d(3, nfc//4, 4, 2, 1, bias=False),nn.LeakyReLU(0.2,inplace=True))
        self.sf_128 = nn.Sequential(nn.Conv2d(nfc//4, nfc//2, 4, 2, 1, bias=False),nn.BatchNorm2d(nfc//2),nn.LeakyReLU(0.1,inplace=True)) 
        self.sf_64 = nn.Sequential(nn.Conv2d(nfc//2, nfc, 4, 2, 1, bias=False),nn.BatchNorm2d(nfc),nn.LeakyReLU(0.1,inplace=True)) 
        
        self.sf_32 = nn.Sequential(nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False), ResBlkE(nfc*2))
        self.sf_16 = nn.Sequential(nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False), ResBlkE(nfc*4))
        self.sf_8 = nn.Sequential(nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False), ResBlkE(nfc*8))
        
        self.sfv_32 = nn.Sequential( nn.AdaptiveAvgPool2d(output_size=4), nn.Conv2d(nfc*2, nfc*2, 4, 1, 0, bias=False), Squeeze() )
        self.sfv_16 = nn.Sequential( nn.AdaptiveAvgPool2d(output_size=4), nn.Conv2d(nfc*4, nfc*4, 4, 1, 0, bias=False), Squeeze() )
        self.sfv_8 = nn.Sequential( nn.AdaptiveAvgPool2d(output_size=4), nn.Conv2d(nfc*8, nfc*8, 4, 1, 0, bias=False), Squeeze() )

        self.nbr_cls = nbr_cls
        self.final_cls = None

    def reset_cls(self):
        if self.final_cls is None:
            self.final_cls = nn.Sequential(nn.LeakyReLU(0.1), nn.Linear(self.nfc*8, self.nbr_cls))
        stdv = 1. / math.sqrt(self.final_cls[1].weight.size(1))
        self.final_cls[1].weight.data.uniform_(-stdv, stdv)
        if self.final_cls[1].bias is not None:
            self.final_cls[1].bias.data.uniform_(-0.1*stdv, 0.1*stdv)

    def get_feats(self, image):
        feat = self.sf_256(image)
        feat = self.sf_128(feat)
        feat = self.sf_64(feat)
        feat_32 = self.sf_32(feat)
        feat_16 = self.sf_16(feat_32)
        feat_8 = self.sf_8(feat_16)
        
        feat_32 = self.sfv_32(feat_32)
        feat_16 = self.sfv_16(feat_16)
        feat_8 = self.sfv_8(feat_8)

        return feat_32, feat_16, feat_8

    def forward(self, image):
        feat_32, feat_16, feat_8 = self.get_feats(image)

        pred_cls = self.final_cls(feat_8)
        return [feat_32, feat_16, feat_8], pred_cls


class ContentEncoder(nn.Module):
    def __init__(self, nfc=64):
        super().__init__()

        self.cf_256 = nn.Sequential(nn.Conv2d(1, nfc//4, 4, 2, 1, bias=False),nn.LeakyReLU(0.2,inplace=True))
        self.cf_128 = nn.Sequential(nn.Conv2d(nfc//4, nfc//2, 4, 2, 1, bias=False),nn.BatchNorm2d( nfc//2),nn.LeakyReLU(0.1,inplace=True)) 
        self.cf_64 = nn.Sequential(nn.Conv2d( nfc//2, nfc, 4, 2, 1, bias=False),nn.BatchNorm2d(nfc),nn.LeakyReLU(0.1,inplace=True)) 
        
        self.cf_32 = nn.Sequential(nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False), ResBlkE(nfc*2))
        self.cf_16 = nn.Sequential(nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False), ResBlkE(nfc*4))
        self.cf_8 = nn.Sequential(nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False), ResBlkE(nfc*8))
        
    def get_feats(self, image):
        feat = self.cf_256(image)
        feat = self.cf_128(feat)
        feat = self.cf_64(feat)
        feat_32 = self.cf_32(feat)
        feat_16 = self.cf_16(feat_32)
        feat_8 = self.cf_8(feat_16)

        return feat_32, feat_16, feat_8

    def forward(self, image):
        feat_32, feat_16, feat_8 = self.get_feats(image)
        return [feat_32, feat_16, feat_8]


def up_decoder(ch_in, ch_out):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ch_in, ch_out*2, 3, 1, 1, bias=False),
        nn.InstanceNorm2d( ch_out*2 ), GLU())


class Decoder(nn.Module):
    def __init__(self, nfc=64):
        super().__init__()
 
        self.base_feat = nn.Parameter(torch.randn(1, nfc*8, 8, 8).normal_(0, 1), requires_grad=True)
        
        self.dmi_8 = DMI(nfc*8)
        self.dmi_16 = DMI(nfc*4)

        self.feat_8_1 = nn.Sequential( ResBlkG(nfc*16), nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(nfc*16, nfc*8, 3, 1, 1, bias=False), nn.InstanceNorm2d(nfc*8) )
        self.feat_8_2 = nn.Sequential( nn.LeakyReLU(0.1,inplace=True), ResBlkG(nfc*8) )
        self.feat_16  = nn.Sequential( nn.LeakyReLU(0.1,inplace=True), nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(nfc*8, nfc*4, 3, 1, 1, bias=False), ResBlkG(nfc*4) )
        self.feat_32  = nn.Sequential( nn.LeakyReLU(0.1,inplace=True), nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(nfc*8, nfc*2, 3, 1, 1, bias=False), ResBlkG(nfc*2) )
        self.feat_64  = nn.Sequential( nn.LeakyReLU(0.1,inplace=True), up_decoder(nfc*4, nfc) ) 
        self.feat_128 = up_decoder(nfc*1, nfc//2)
        self.feat_256 = up_decoder(nfc//2, nfc//4)
        self.feat_512 = up_decoder(nfc//4, nfc//8)
        
        self.to_rgb = nn.Sequential( nn.Conv2d(nfc//8, 3, 3, 1, 1, bias=False), nn.Tanh() )
        
        self.style_8 = nn.Sequential( nn.Linear(nfc*8, nfc*8), nn.ReLU(), nn.Linear(nfc*8, nfc*8), nn.BatchNorm1d(nfc*8), UnSqueeze() )
        self.style_64 = nn.Sequential( nn.Linear(nfc*8, nfc), nn.ReLU(), nn.Linear(nfc, nfc), nn.Sigmoid() , UnSqueeze())
        self.style_128 = nn.Sequential( nn.Linear(nfc*4, nfc//2), nn.ReLU(), nn.Linear(nfc//2, nfc//2), nn.Sigmoid() , UnSqueeze())
        self.style_256 = nn.Sequential( nn.Linear(nfc*2, nfc//4), nn.ReLU(), nn.Linear(nfc//4, nfc//4), nn.Sigmoid() , UnSqueeze())

    def forward(self, content_feats, style_vectors):

        feat_8 = self.feat_8_1( torch.cat( [content_feats[2], self.base_feat.repeat(style_vectors[0].shape[0], 1, 1, 1)], dim=1 ) )            
        feat_8 = self.dmi_8(feat_8, content_feats[2])

        bs = feat_8.shape[0]

        feat_8 = feat_8 * self.style_8( style_vectors[2] )
        feat_8 = self.feat_8_2(feat_8)

        feat_16 = self.feat_16(feat_8) 
        feat_16 = self.dmi_16(feat_16, content_feats[1])
        feat_16 = torch.cat([feat_16, content_feats[1]], dim=1)

        feat_32 = self.feat_32(feat_16) 
        feat_32 = torch.cat([feat_32, content_feats[0]], dim=1)

        feat_64 = self.feat_64(feat_32) * self.style_64(style_vectors[2]) 
        feat_128 = self.feat_128(feat_64) * self.style_128(style_vectors[1]) 
        feat_256 = self.feat_256(feat_128) * self.style_256(style_vectors[0]) 
        feat_512 = self.feat_512(feat_256) 

        return self.to_rgb(feat_512)


class AE(nn.Module):
    def __init__(self, nfc, nbr_cls=500):
        super().__init__()  

        self.style_encoder = StyleEncoder(nfc, nbr_cls=nbr_cls)
        self.content_encoder = ContentEncoder(nfc)
        self.decoder = Decoder(nfc)

    @torch.no_grad()
    def forward(self, skt_img, style_img):
        style_feats = self.style_encoder.get_feats( F.interpolate(style_img, size=512) )
        content_feats = self.content_encoder( F.interpolate( skt_img , size=512) )
        gimg = self.decoder(content_feats, style_feats)
        return gimg, style_feats

    def load_state_dicts(self, path):
        ckpt = torch.load(path)
        self.style_encoder.reset_cls()
        self.style_encoder.load_state_dict(ckpt['s'])
        self.content_encoder.load_state_dict(ckpt['c'])
        self.decoder.load_state_dict(ckpt['d'])
        print('AE load success')

def down_gan(ch_in, ch_out):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)),
        nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2, inplace=True))


def up_gan(ch_in, ch_out):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        spectral_norm( nn.Conv2d(ch_in, ch_out*2, 3, 1, 1, bias=False) ),
        nn.BatchNorm2d( ch_out*2 ), NoiseInjection(ch_out*2), GLU())


def repeat_upscale(feat, scale_factor=2):
    feat = feat.repeat(1,1,scale_factor,scale_factor)
    return feat


class RefineGenerator_art(nn.Module):
    def __init__(self, nfc=64, im_size=512):
        super().__init__()  

        self.im_size = im_size

        d16, d32, d64, d128, d256, d512 = nfc*8, nfc*8, nfc*4, nfc*2, nfc, nfc//2 

        self.from_noise_32 = nn.Sequential( UnSqueeze(),
            spectral_norm(nn.ConvTranspose2d(nfc*8, nfc*8, 4, 1, 0, bias=False)), #4
            nn.BatchNorm2d(nfc*8), GLU(), up_gan(nfc*4, nfc*2),  up_gan(nfc*2, nfc*2), up_gan(nfc*2, nfc*1)) #32 

        self.from_style = nn.Sequential( UnSqueeze(),
            spectral_norm(nn.ConvTranspose2d(nfc*(8+4+2), nfc*16, 4, 1, 0, bias=False)), #4
            nn.BatchNorm2d(nfc*16), GLU(), up_gan(nfc*8, nfc*4) )
        
        self.encode_256 = nn.Sequential( spectral_norm(nn.Conv2d(3, d256, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2,inplace=True))
        self.encode_128 = down_gan(d256, d128)
        self.encode_64 = down_gan(d128, d64)
        self.encode_32 = down_gan(d64, d32)
        self.encode_16 = down_gan(d32, d16)

        self.residual_16 = nn.Sequential( ResBlkG(d16+nfc*4), Swish(), ResBlkG(d16+nfc*4), Swish() )

        self.decode_32  = up_gan(d16+nfc*4, d32)
        self.decode_64  = up_gan(d32+nfc, d64) 
        self.decode_128 = up_gan(d64, d128)
        self.decode_256 = up_gan(d128, d256)
        self.decode_512 = up_gan(d256, d512)
        if im_size == 1024:
            self.decode_1024 = up_gan(d512, nfc//4)

        self.style_64  =  nn.Sequential( spectral_norm( nn.Linear(nfc*8, d64) ), nn.ReLU(), nn.Linear(d64, d64),  nn.Sigmoid(), UnSqueeze())
        self.style_128 =  nn.Sequential( spectral_norm( nn.Linear(nfc*8, d128)), nn.ReLU(), nn.Linear(d128, d128),nn.Sigmoid(), UnSqueeze())
        self.style_256 =  nn.Sequential( spectral_norm( nn.Linear(nfc*4, d256)), nn.ReLU(), nn.Linear(d256, d256),nn.Sigmoid(), UnSqueeze())
        self.style_512 =  nn.Sequential( spectral_norm( nn.Linear(nfc*2, d512)), nn.ReLU(), nn.Linear(d512, d512),nn.Sigmoid(), UnSqueeze())
        
        self.to_rgb = nn.Sequential( nn.Conv2d(nfc//2, 3, 3, 1, 1, bias=False), nn.Tanh() )
        if im_size == 1024:
            self.to_rgb = nn.Sequential( nn.Conv2d(nfc//4, 3, 3, 1, 1, bias=False), nn.Tanh() )
        
        if DATA_NAME=='shoe':
            self.bs_0 = nn.Parameter(torch.randn(1, nfc*2))
            self.bs_1 = nn.Parameter(torch.randn(1, nfc*4))
            self.bs_2 = nn.Parameter(torch.randn(1, nfc*8))

    def forward(self, image, style_vectors):
         
        s_16 = repeat_upscale( self.from_style(torch.cat(style_vectors,1)), scale_factor=2 )
        if DATA_NAME=='shoe':  
            s_16 = torch.zeros_like(s_16)
            
        n_32 = self.from_noise_32(torch.randn_like(style_vectors[2]))

        e_256 = self.encode_256( image )
        e_128 = self.encode_128( e_256 )
        e_64 = self.encode_64( e_128 )
        e_32 = self.encode_32( e_64 )
        e_16 = self.encode_16(e_32)

        e_16 = self.residual_16( torch.cat([e_16, s_16],dim=1) )
        
        d_32 = self.decode_32( e_16 )
        d_64 = self.decode_64( torch.cat([d_32, n_32], dim=1) ) 
        if DATA_NAME!='shoe':
            d_64 *= self.style_64(style_vectors[2])
        d_128 = self.decode_128( d_64 + e_64 ) 
        if DATA_NAME!='shoe':
            d_128 *= self.style_128(style_vectors[2])
        d_256 = self.decode_256( d_128 + e_128 )
        if DATA_NAME!='shoe':
            d_256 *= self.style_256(style_vectors[1])
        d_512 = self.decode_512( d_256 + e_256 ) 
        if DATA_NAME!='shoe':
            d_512 *= self.style_512(style_vectors[0])
        
        if self.im_size == 1024:
            d_final = self.decode_1024(d_512)
        else:
            d_final = d_512
        return self.to_rgb(d_final)


class RefineGenerator_face(nn.Module):
    def __init__(self, nfc, im_size):
        super().__init__()  

        self.im_size = im_size

        e1, e2, e3, e4 = 16, 32, 64, 128
        self.encode_1 = down_gan(3, e1)      #256
        self.encode_2 = down_gan(e1, e2)     #128
        self.encode_3 = down_gan(e2, e3)     #64
        self.encode_4 = down_gan(e3, e4)    #32

        s1, s2, s3, s4 = 256, 128, 128, 64
        self.style = nn.Sequential(nn.Linear(nfc*(8+4+2), 512), nn.LeakyReLU(0.1))
        self.from_style_32 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(512, s1, 4, 1, 0, bias=False)), #4
            nn.BatchNorm2d(s1), GLU(), up_gan(s1//2, s2), up_gan(s2, s3), up_gan(s3, s4)) #32

        d1, d2, d3, d4, d5 = 256, 128, 64, 32, 16
        self.decode_64 = up_gan( e4 + s4 , d1)
        self.decode_128 = up_gan(d1+e3, d2)
        self.decode_256 = up_gan(d2+e2, d3)
        self.decode_512 = up_gan(d3+e1, d4)
        if im_size == 1024:
            self.decode_1024 = up_gan(d4, d5)

        self.style_blocks = nn.ModuleList()
        
        chs = [d1, d2, d3, d4]
        if im_size == 1024:
            chs.append(d5)
        for i in range(len(chs)):
            self.style_blocks.append(nn.Sequential( 
                    nn.Linear(512, chs[i]), nn.ReLU(), nn.Linear(chs[i], chs[i]), nn.Sigmoid() ))

        self.final = nn.Sequential( spectral_norm( 
                            nn.Conv2d(d4, 3, 3, 1, 1, bias=False) ), nn.Tanh() )
        if im_size == 1024:
            self.final = nn.Sequential( spectral_norm( 
                            nn.Conv2d(d5, 3, 3, 1, 1, bias=False) ), nn.Tanh() )
        
    def forward(self, image, style):
        e_256 = self.encode_1( image )
        e_128 = self.encode_2( e_256 )
        e_64 = self.encode_3( e_128 )
        e_32 = self.encode_4( e_64 )

        style = self.style(torch.cat(style, dim=1))
        s_32 = self.from_style_32( style.unsqueeze(-1).unsqueeze(-1) )
        
        if random.randint(0, 1) == 1:
            s_32 = s_32.flip(2)
        if random.randint(0, 1) == 1:
            s_32 = s_32.flip(3)
        
        feat_64 = self.decode_64( torch.cat([e_32, s_32], dim=1) ) * self.style_blocks[0](style).unsqueeze(-1).unsqueeze(-1)
        feat_128 = self.decode_128( torch.cat([e_64, feat_64], dim=1) ) * self.style_blocks[1](style).unsqueeze(-1).unsqueeze(-1)
        feat_256 = self.decode_256( torch.cat([e_128, feat_128], dim=1) ) * self.style_blocks[2](style).unsqueeze(-1).unsqueeze(-1)
        feat_512 = self.decode_512( torch.cat([e_256, feat_256], dim=1) ) * self.style_blocks[3](style).unsqueeze(-1).unsqueeze(-1)
        if self.im_size == 1024:
            feat_1024 = self.decode_1024( feat_512 ) * self.style_blocks[4](style).unsqueeze(-1).unsqueeze(-1)
            return self.final(feat_1024)
        else:
            return self.final(feat_512)


class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_skip=0):
        super().__init__()

        self.ch_out = ch_out
        self.down_main = nn.Sequential(
                spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=False)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True)
                )

        self.skip = False
        
        if ch_skip > 0:  
            self.skip = True 
            self.skip_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                spectral_norm( nn.Conv2d(ch_skip, ch_out, 4, 1, 0, bias=False) ),
                nn.ReLU(),
                spectral_norm( nn.Conv2d(ch_out, ch_out*2, 1, 1, 0, bias=False) ),
            ) 

    def forward(self, feat, skip_feat=None):
        feat_out = self.down_main(feat) 
        if skip_feat is not None and self.skip:
            addon = self.skip_conv(skip_feat)
            feat_out = feat_out * torch.sigmoid(addon[:,:self.ch_out]) + torch.tanh(addon[:,self.ch_out:])        
        
        return feat_out


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        modules = [
            nn.Sequential(spectral_norm(nn.Conv2d(nc, ndf//4, 4, 2, 1, bias=False)),
                          nn.LeakyReLU(0.2, inplace=True)),
            DownBlock(ndf//4, ndf//2),
            DownBlock(ndf//2, ndf*1),
            DownBlock(ndf*1,  ndf*2),
            DownBlock(ndf*2,  ndf*4, ch_skip=ndf//4),
            ]

        if im_size == 512:
            modules.append(
                DownBlock(ndf*4,  ndf*16, ch_skip=ndf//2),
            )
        elif im_size == 1024:
            modules.append(
                DownBlock(ndf*4,  ndf*8, ch_skip=ndf//2))
            modules.append(
                DownBlock(ndf*8,  ndf*16, ch_skip=ndf*1),
            )
        modules.append(
                        nn.Sequential(
                            spectral_norm(nn.Conv2d(ndf*16, ndf*16, 1, 1, 0, bias=False)),
                            nn.BatchNorm2d(ndf*16),
                            nn.LeakyReLU(0.2, inplace=True),
                            spectral_norm(nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False)))
                       )

        self.main = nn.ModuleList(modules)
        
        self.apply(weights_init)


    def forward(self, x):
        # x shape 512
        feat_256 = self.main[0](x)
        feat_128 = self.main[1](feat_256)
        feat_64 = self.main[2](feat_128)
        feat_32 = self.main[3](feat_64)

        feat_16 = self.main[4](feat_32, feat_256)
        feat_8 = self.main[5](feat_16, feat_128)
        if self.im_size == 1024:
            feat_last = self.main[6](feat_8, feat_64)
        else:
            feat_last = feat_8

        return self.main[-1](feat_last)