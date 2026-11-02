from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass
class ModelConfig:
    kind: str = "vae"
    image_size: int = 192
    latent_dim: int = 64
    channels: tuple = (16,32,64,128)


class ReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config,dict): config=ModelConfig(**config)
        if config.kind not in ("ae","vae") or config.image_size % 2**len(config.channels):
            raise ValueError("Invalid model kind or image size")
        self.config=config
        layers=[]; previous=1
        for channel in config.channels:
            layers += [nn.Conv2d(previous,channel,3,2,1),nn.GroupNorm(min(8,channel),channel),nn.SiLU()]
            previous=channel
        self.encoder=nn.Sequential(*layers)
        self.side=config.image_size//2**len(config.channels)
        flattened=previous*self.side*self.side
        self.mean=nn.Linear(flattened,config.latent_dim)
        self.logvar=nn.Linear(flattened,config.latent_dim) if config.kind=="vae" else None
        self.expand=nn.Linear(config.latent_dim,flattened)
        decoder=[]
        for channel in list(config.channels[-2::-1])+[16]:
            decoder += [nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
                        nn.Conv2d(previous,channel,3,padding=1),nn.GroupNorm(min(8,channel),channel),nn.SiLU()]
            previous=channel
        decoder += [nn.Conv2d(previous,1,3,padding=1),nn.Sigmoid()]
        self.decoder=nn.Sequential(*decoder)

    def forward(self,x,sample=None):
        if x.shape[1:]!=(1,self.config.image_size,self.config.image_size):
            raise ValueError("Input geometry does not match checkpoint")
        features=self.encoder(x).flatten(1)
        mean=self.mean(features)
        logvar=self.logvar(features).clamp(-10,6) if self.logvar is not None else torch.zeros_like(mean)
        if sample is None: sample=self.training and self.config.kind=="vae"
        latent=mean+torch.exp(.5*logvar)*torch.randn_like(mean) if sample and self.logvar is not None else mean
        result=self.decoder(self.expand(latent).view(-1,self.config.channels[-1],self.side,self.side))
        return result,mean,logvar

    def configuration(self): return asdict(self.config)


def losses(reconstructed,original,brain,mean,logvar,beta=0.,kind="vae",reconstruction_loss="l1",background_weight=.05):
    error=(reconstructed-original).abs()
    valid=brain.flatten(1).sum(1).clamp_min(1)
    reconstruction=((error*brain).flatten(1).sum(1)/valid).mean()
    if reconstruction_loss not in ("l1","mse"):raise ValueError("Unknown reconstruction loss")
    objective=reconstruction if reconstruction_loss=="l1" else ((error.square()*brain).flatten(1).sum(1)/valid).mean()
    background=((error*(1-brain)).flatten(1).mean(1)).mean()
    # KL averaged over batch AND latent dimensions. Beta is defined for this
    # explicit convention and is not interchangeable with summed-latent beta.
    kl=(-.5*(1+logvar-mean.square()-logvar.exp())).mean() if kind=="vae" else mean.sum()*0
    return objective+background_weight*background+beta*kl,reconstruction,kl
