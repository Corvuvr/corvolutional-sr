import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision

from model.modules import *


####################################
# DEFINE MODEL INSTANCES
####################################
    
   
class RT4KSR_Rep(nn.Module):
    def __init__(self,
                 num_channels, 
                 num_feats, 
                 num_blocks,
                 upscale,
                 act,
                 eca_gamma,
                 is_train,
                 forget,
                 layernorm,
                 residual) -> None:
        super().__init__()
        self.forget = forget
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        
        r: int = 2
        self.down = nn.PixelUnshuffle(r)
        self.up = nn.PixelShuffle(r)
        self.head = nn.Sequential(nn.Conv2d(num_channels * (r**2), num_feats, 3, padding=1))
        self.head_flow = nn.Sequential(nn.Conv2d(2 * (r**2), num_feats, 3, padding=1))
        
        hfb = []
        if is_train:
            hfb.append(ResBlock(num_feats, ratio=2))
        else:
            hfb.append((RepResBlock(num_feats)))
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        body = []
        for i in range(num_blocks):
            if is_train:
                body.append(SimplifiedNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm, residual=residual))
            else:
                body.append(SimplifiedRepNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm, residual=residual))
        
        self.body = nn.Sequential(*body)
        
        tail = [LayerNorm2d(num_feats)]
        if is_train:
            tail.append(ResBlock(num_feats, ratio=2))
        else:
            tail.append(RepResBlock(num_feats))
        self.tail = nn.Sequential(*tail)
                    
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * ((2 * upscale) ** 2), 3, padding=1),
            nn.PixelShuffle(upscale*2)            
        )
        
    def forward(self, x, y):
        FLOW_HEAD: bool = True
        # Low Res Image: [1, 3, 218, 512] - RGB
        # Flow:          [1, 2, 218, 512] - dx/dy
        lr_img = x
        flow   = y
        hf = lr_img - self.gaussian(lr_img)
        hf_flo = flow - self.gaussian(flow)

        # Unshuffle to save computation
        x_unsh  = self.down(lr_img)
        hf_unsh = self.down(hf)
        hf_flo_unsh = self.down(hf_flo)

        # RuntimeError: Given groups=1, weight of size [24, 12, 3, 3], expected input[1, 8, 109, 256] to have 12 channels, but got 8 channels instead
        shallow_feats_hf = self.head(hf_unsh)        
        shallow_feats_hf_flo = self.head_flow(hf_flo_unsh)        
        shallow_feats_lr = self.head(x_unsh)

        # stage 2            
        deep_feats = self.body(shallow_feats_lr)
        if FLOW_HEAD:
            hf_feats   = self.hfb(shallow_feats_hf + shallow_feats_hf_flo)
        else:
            hf_feats   = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)        
        return out
    
    
####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################

def rt4ksr_rep(config):
    act = activation(config.act_type)
    model = RT4KSR_Rep(num_channels=3, 
                       num_feats=config.feature_channels, 
                       num_blocks=config.num_blocks, 
                       upscale=config.scale,
                       act=act,
                       eca_gamma=0,
                       forget=False,
                       is_train=config.is_train,
                       layernorm=True,
                       residual=False)
    return model