"""Experimental spatial denoising AE/VAE; separate from frozen vector models."""
from dataclasses import asdict,dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RestorationConfig:
    kind:str='vae'
    image_size:int=192
    base_channels:int=8
    latent_channels:int=8
    family:str='spatial_denoising_v1'


def block(inputs,outputs):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,padding=1),nn.GroupNorm(min(8,outputs),outputs),nn.SiLU(),
                         nn.Conv2d(outputs,outputs,3,padding=1),nn.GroupNorm(min(8,outputs),outputs),nn.SiLU())


class SpatialRestorationModel(nn.Module):
    """Spatial bottleneck and feature skips; never add the raw input to output.

    Skips can still transmit pathology. Only experiments can establish whether
    healthy-only coarse-noise restoration improves anomaly discrimination.
    """
    def __init__(self,config):
        super().__init__()
        if isinstance(config,dict):config=RestorationConfig(**config)
        if (config.kind not in ('ae','vae') or config.image_size%8 or config.base_channels<8 or
                config.base_channels%8 or config.latent_channels<1 or config.family!='spatial_denoising_v1'):
            raise ValueError('Invalid spatial restoration configuration')
        self.config=config;c=config.base_channels
        self.encoder=nn.ModuleList([block(1,c),block(c,2*c),block(2*c,4*c),block(4*c,8*c)])
        self.mean=nn.Conv2d(8*c,config.latent_channels,1)
        self.logvar=nn.Conv2d(8*c,config.latent_channels,1) if config.kind=='vae' else None
        self.expand=nn.Conv2d(config.latent_channels,8*c,1)
        self.decoder=nn.ModuleList([block(12*c,4*c),block(6*c,2*c),block(3*c,c)])
        self.output=nn.Conv2d(c,1,1)

    def encode(self,x):
        if x.ndim!=4 or x.shape[1:]!=(1,self.config.image_size,self.config.image_size):
            raise ValueError('Input geometry does not match spatial checkpoint')
        skips=[]
        for level,encoder in enumerate(self.encoder):
            if level:x=F.avg_pool2d(x,2)
            x=encoder(x)
            if level<3:skips.append(x)
        mean=self.mean(x)
        logvar=self.logvar(x).clamp(-10,6) if self.logvar is not None else torch.zeros_like(mean)
        return mean,logvar,skips

    def decode(self,latent,skips,skip_scale=1.):
        x=self.expand(latent)
        for decoder,skip in zip(self.decoder,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=decoder(torch.cat([x,skip_scale*skip],dim=1))
        return torch.sigmoid(self.output(x))

    def forward(self,x,sample=None):
        mean,logvar,skips=self.encode(x)
        if sample is None:sample=self.training and self.config.kind=='vae'
        latent=mean+torch.exp(.5*logvar)*torch.randn_like(mean) if sample and self.logvar is not None else mean
        return self.decode(latent,skips),mean,logvar

    def configuration(self):return asdict(self.config)


def load_restoration(checkpoint):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SpatialRestorationModel(payload['model_config']);model.load_state_dict(payload['model']);model.eval()
    return model
