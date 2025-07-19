import torch
from torch import nn
import torch.nn.functional as F
from .grad_rev_layer import GradReverse

class DiscriminatorProposal(nn.Module):
    def __init__(self, in_features: int):
        super(DiscriminatorProposal, self).__init__()
        self.net = nn.Linear(in_features, 1, bias=False).cuda()

        '''
        if the loss function diverges, try to implement a more complex network as follows:
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, 1, bias=False)
        ).cuda()
        '''


    def forward(self, x: torch.Tensor, domain_target: bool = False, alpha: float = 1.0):
        x = GradReverse.apply(x, alpha)        
        logits = self.net(x)                    

        target = torch.ones_like(logits).cuda() if domain_target else torch.zeros_like(logits).cuda()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

        return {
            "loss_instance_d": loss,
            "logits": logits
        }

class DiscriminatorProposalDC5(nn.Module):
    def __init__(self, in_channels: int):
        super(DiscriminatorProposalDC5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) 
        ).cuda()
        self.classifier = nn.Linear(256, 1, bias=False).cuda()

    def forward(self, x: torch.Tensor, domain_target: bool = False, alpha: float = 1.0):
        x = GradReverse.apply(x, alpha)           
        x = self.conv(x)                          
        x = torch.flatten(x, start_dim=1)         
        logits = self.classifier(x)               

        target = torch.ones_like(logits).cuda() if domain_target else torch.zeros_like(logits).cuda()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

        return {
            "loss_instance_d": loss,
            "logits": logits
        }