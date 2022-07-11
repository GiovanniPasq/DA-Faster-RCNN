import torch
from torch import nn
from .grad_rev_layer import *
from fvcore.nn import sigmoid_focal_loss_jit

class DiscriminatorProposal(nn.Module):
    def __init__(self, in_feature):
        super(DiscriminatorProposal, self).__init__()
        self.reducer = nn.Sequential(
            nn.Linear(in_feature, in_feature, bias = False),  
            nn.ReLU(inplace=True),
            nn.Linear(in_feature, 1, bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_s, alpha=0.25,gamma=2,reduction="mean")
        return {"loss_instance_d": loss}

class DiscriminatorProposalDC5(nn.Module):
    def __init__(self, in_feature):
        super(DiscriminatorProposalDC5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_feature, 1, kernel_size=(1, 1), bias = False)
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
        return {"loss_instance_d": loss}