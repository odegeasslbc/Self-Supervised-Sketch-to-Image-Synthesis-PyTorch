import functools
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cat, sigmoid
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from torch.jit import ScriptModule, script_method, trace

#####################################################################
#####   functions
#####################################################################

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def get_batched_gram_matrix(input):
    # take a batch of features: B X C X H X W
    # return gram of each image: B x C x C
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(2,1)) 
    return G.div(b * c * d)
    
class Adaptive_pool(nn.Module):
    '''
    take a input tensor of size: B x C' X C'
    output a maxpooled tensor of size: B x C x H x W
    '''
    def __init__(self, channel_out, hw_out):
        super().__init__()
        self.channel_out = channel_out
        self.hw_out = hw_out
        self.pool = nn.AdaptiveAvgPool2d((channel_out, hw_out**2))
    def forward(self, input):
        if len(input.shape) == 3:
            input.unsqueeze_(1)
        return self.pool(input).view(-1, self.channel_out, self.hw_out, self.hw_out)
### new function

#####################################################################
#####   models
#####################################################################
class VGGSimple(nn.Module):
    def __init__(self):
        super(VGGSimple, self).__init__()

        self.features = self.make_layers()
        
        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, img, after_relu=True, base=4):
        # re-normalize from [-1, 1] to [0, 1] then to the range used for vgg
        feat = (((img+1)*0.5) - self.norm_mean.to(img.device)) / self.norm_std.to(img.device)
        # the layer numbers used to extract features
        cut_points = [2, 7, 14, 21, 28]
        if after_relu:
            cut_points = [c+2 for c in cut_points]
        for i in range(31):
            feat = self.features[i](feat)
            if i == cut_points[0]:
                feat_64 = F.adaptive_avg_pool2d(feat, base*16)
            if i == cut_points[1]:
                feat_32 = F.adaptive_avg_pool2d(feat, base*8)
            if i == cut_points[2]:
                feat_16 = F.adaptive_avg_pool2d(feat, base*4)
            if i == cut_points[3]:
                feat_8 = F.adaptive_avg_pool2d(feat, base*2)
            if i == cut_points[4]:
                feat_4 = F.adaptive_avg_pool2d(feat, base)
        
        return feat_64, feat_32, feat_16, feat_8, feat_4

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)


# this model is used for pre-training
class VGG_3label(nn.Module):
    def __init__(self, nclass_artist=1117, nclass_style=55, nclass_genre=26):
        super(VGG_3label, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier_feat = self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512))

        self.classifier_style = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_style))
        self.classifier_genre = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_genre))
        self.classifier_artist = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_artist))

        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    
        self.avgpool_4 = nn.AdaptiveAvgPool2d((4, 4))
        self.avgpool_8 = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool_16 = nn.AdaptiveAvgPool2d((16, 16))
    
    def get_features(self, img, after_relu=True, base=4):
        feat = (((img+1)*0.5) - self.norm_mean.to(img.device)) / self.norm_std.to(img.device)
        cut_points = [2, 7, 14, 21, 28]
        if after_relu:
            cut_points = [4, 9, 16, 23, 30]
        for i in range(31):
            feat = self.features[i](feat)
            if i == cut_points[0]:
                feat_64 = F.adaptive_avg_pool2d(feat, base*16)
            if i == cut_points[1]:
                feat_32 = F.adaptive_avg_pool2d(feat, base*8)
            if i == cut_points[2]:
                feat_16 = F.adaptive_avg_pool2d(feat, base*4)
            if i == cut_points[3]:
                feat_8 = F.adaptive_avg_pool2d(feat, base*2)
            if i == cut_points[4]:
                feat_4 = F.adaptive_avg_pool2d(feat, base)
        #feat_code = self.classifier_feat(self.avgpool(feat).view(img.size(0), -1))
        return feat_64, feat_32, feat_16, feat_8, feat_4#, feat_code


    def load_pretrain_weights(self):
        pretrained_vgg16 = vgg.vgg16(pretrained=True)
        self.features.load_state_dict(pretrained_vgg16.features.state_dict())
        self.classifier_feat[0] = pretrained_vgg16.classifier[0] 
        self.classifier_feat[3] = pretrained_vgg16.classifier[3] 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, img):
        feature = self.classifier_feat( self.avgpool(self.features(img)).view(img.size(0), -1) )
        pred_style = self.classifier_style(feature)
        pred_genre = self.classifier_genre(feature)
        pred_artist = self.classifier_artist(feature)
        return pred_style, pred_genre, pred_artist


class UnFlatten(nn.Module):
    def __init__(self, block_size):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 0, bias=True)),
            norm_layer(out_channel), 
            nn.LeakyReLU(0.01), 
            )

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        return self.main(y)


class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d, down=True):
        super().__init__()

        m = [   spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)),
                norm_layer(out_channel), 
                nn.LeakyReLU(0.1) ]
        if down:
            m.append(nn.AvgPool2d(2, 2))
        self.main = nn.Sequential(*m)

    def forward(self, x):
        return self.main(x)




class Generator(nn.Module):
    def __init__(self, infc=512, nfc=64, nc_out=3):
        super(Generator, self).__init__()

        self.decode_32 = UpConvBlock(infc, nfc*4)	#32
        self.decode_64 = UpConvBlock(nfc*4, nfc*4)    #64
        self.decode_128 = UpConvBlock(nfc*4, nfc*2)    #128

        self.final = nn.Sequential(
            spectral_norm( nn.Conv2d(nfc*2, nc_out, 3, 1, 1, bias=True) ),
            nn.Tanh())

    def forward(self, input):

        decode_32 = self.decode_32(input)
        decode_64 = self.decode_64(decode_32)
        decode_128 = self.decode_128(decode_64)

        output = self.final(decode_128)
        return output


class Discriminator(nn.Module):
    def __init__(self, nfc=512, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            DownConvBlock(nfc, nfc//2, norm_layer=norm_layer, down=False),
            DownConvBlock(nfc//2, nfc//4, norm_layer=norm_layer), #4x4
            spectral_norm( nn.Conv2d(nfc//4, 1, 4, 2, 0) )
        )
	
    def forward(self, input):
        out = self.main(input)
        return out.view(-1)

