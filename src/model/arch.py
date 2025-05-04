import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision

from model.modules import *


####################################
# DEFINE MODEL INSTANCES
####################################
    
   
class CorvolutionalSuperResolution(nn.Module):
    def __init__(self,
                 num_channels, 
                 num_feats, 
                 num_blocks,
                 upscale,
                 act,
                 eca_gamma,
                 is_train,
                 layernorm,
                 residual,
                 forward: str = 'vanilla'
                 ) -> None:
        super().__init__()
        self.forward_option = forward
        print(f"Running {self.forward_option}...")
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_flow = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        
        r: int = 2
        self.down = nn.PixelUnshuffle(r)
        self.up = nn.PixelShuffle(r)
        # Different Conv Layers
        self.head        = nn.Sequential(nn.Conv2d(num_channels * (r**2), num_feats, 3, padding=1))
        self.head_flow   = nn.Sequential(nn.Conv2d(2 * (r**2), num_feats, 3, padding=1))
        self.head_common = nn.Sequential(nn.Conv2d(5 * (r**2), num_feats, 3, padding=1))
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
        match self.forward_option:
            case 'vanilla':
                return self.forward_vanilla(x,y)
            case 'vanilla_hf':
                return self.forward_vanilla_hf(x,y)
            case 'flow':
                return self.forward_flow(x,y)
            case 'flow_cat':
                return self.forward_flow_cat(x,y)
            case _:
                return self.forward_vanilla(x,y)


    def forward_vanilla(self, x, y):
        lr_img = x 
        # Unshuffle
        x_unsh  = self.down(lr_img)
        # Conv
        shallow_feats_lr = self.head(x_unsh)
        # NAF          
        deep_feats = self.body(shallow_feats_lr)
        # ResBlock
        deep_feats = self.tail(deep_feats)
        # Conv + Shuffle
        out = self.upsample(deep_feats)        
        return out
    
    def forward_vanilla_hf(self, x, y):
        lr_img = x
        hf = lr_img - self.gaussian(lr_img)                  
        # Unshuffle
        x_unsh  = self.down(lr_img)
        hf_unsh = self.down(hf)
        # Conv
        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)
        # NAF           
        deep_feats = self.body(shallow_feats_lr)
        # HFB
        hf_feats = self.hfb(shallow_feats_hf)
        # ResBlock
        deep_feats = self.tail(
            self.gamma * deep_feats + \
            hf_feats
        )
        # Conv + Shuffle
        out = self.upsample(deep_feats)        
        return out   

    def forward_flow(self, x, y):
        lr_img = x
        flow = y
        # High Frequency Features
        hf = lr_img - self.gaussian(lr_img)            
        ff = flow   - self.gaussian(flow)                    
        # Unshuffle
        x_unsh  = self.down(lr_img)
        hf_unsh = self.down(hf)
        ff_unsh = self.down(ff)
        # Conv
        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)
        shallow_feats_ff = self.head_flow(ff_unsh)
        # NAF
        with torch.no_grad():
            deep_feats = self.body(shallow_feats_lr)
        # HFB
        hf_feats   = self.hfb(shallow_feats_hf)
        ff_feats   = self.hfb(shallow_feats_ff)
        # ResBlock
        deep_feats = self.tail(
            self.gamma * deep_feats + \
            self.gamma_flow * ff_feats + \
            hf_feats
        )
        # Conv + Shuffle
        out = self.upsample(deep_feats)        
        return out  

    def forward_flow_cat(self, x, y):
        lr_img = x
        flow = y
        # Prep Data
        channel_stack = torch.cat((lr_img, flow), dim=1)        # [1, 5,  218, 512]
        hf_stack = channel_stack - self.gaussian(channel_stack) # [1, 5,  218, 512]
        # Unshuffle
        hf_stack_unsh = self.down(hf_stack)                     # [1, 20, 109, 256]
        x_unsh        = self.down(lr_img)
        # Conv
        with torch.no_grad():
            deep_feats_conv = self.head(x_unsh)
        hf_stack_conv    = self.head_common(hf_stack_unsh)      # [1, 24, 109, 256]
        # NAF
        deep_feats = self.body(deep_feats_conv)
        # HFB
        hf_feats = self.hfb(hf_stack_conv)
        # ResBlock
        deep_feats = self.tail(
            self.gamma * deep_feats + \
            self.gamma_flow * hf_feats + \
            hf_feats
        )
        # Conv + Shuffle
        out = self.upsample(deep_feats)        
        return out  

####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################

def corvolutional_rep(config):
    act = activation(config.act_type)
    model = CorvolutionalSuperResolution(num_channels=3, 
                       num_feats=config.feature_channels, 
                       num_blocks=config.num_blocks, 
                       upscale=config.scale,
                       act=act,
                       eca_gamma=0,
                       forward=config.forward_option,
                       is_train=config.is_train,
                       layernorm=True,
                       residual=False)
    return model