import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
 
    def forward(self, inp, target):
        kl_divs = F.kl_div(torch.log(inp), target, 'batchmean')
        return kl_divs