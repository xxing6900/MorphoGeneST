# zinb.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAct(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor, float):
        scale_factor = torch.full((len(mean),), scale_factor).to(x.device)
    scale_factor = scale_factor[:, None]
    
    # 将 Mean 按 Size Factor 缩放
    mean = mean * scale_factor

    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        result += ridge_lambda * torch.square(pi)
        
    return torch.mean(result)