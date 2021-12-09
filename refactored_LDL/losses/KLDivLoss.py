import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.size_average = True
        self.reduce = False
 
    def forward(self, inp, target):
        kl_divs = F.kl_div(torch.log(inp), target, \
        size_average=self.size_average, reduce=self.reduce)
        kl_divs = torch.mean(torch.mean(kl_divs, 1))

        return kl_divs