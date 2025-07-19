import torch
from torch import nn
import torch.nn.functional as F
from .grad_rev_layer import GradReverse

class ImageDomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super(ImageDomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False)  
        ).cuda()

    def forward(self, x: torch.Tensor, domain_target: bool = False, alpha: float = 1.0):
        x = GradReverse.apply(x, alpha)          
        logits = self.net(x)                     

        target = torch.ones_like(logits).cuda() if domain_target else torch.zeros_like(logits).cuda()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

        return {
            "loss_image_d": loss,
            "logits": logits 
        }