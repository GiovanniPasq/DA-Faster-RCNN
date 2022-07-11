import torch
from torch import nn
from .grad_rev_layer import *
from fvcore.nn import sigmoid_focal_loss_jit

class Discriminator(nn.Module):
    def __init__(self, in_feature):
        super(Discriminator, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_feature, int(in_feature/2), kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_feature/2), int(in_feature/4), kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(in_feature/4), 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x, 1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_s, alpha=0.25,gamma=2,reduction="mean")
        return {"loss_image_d": loss}