import numpy as np
import torch

def rep(vec):
    temp = np.array(vec)
    temp = (temp - np.min(temp))/np.ptp(temp)
    return temp.tolist()

def hit_ratio(logit, target):
    shortlist = logit.topk(1000, sorted = True)[1]
    repeated_target = target.unsqueeze(-1).repeat(1, 1000)
    cnt_10 = 0
    cnt_20 = 0
    cnt_30 = 0
    cnt_40 = 0
    cnt_50 = 0 
    cnt_10 += torch.sum(shortlist[:, :10] == repeated_target[:, :10])
    cnt_20   += torch.sum(shortlist[:, :20] == repeated_target[:, :20])
    cnt_30   += torch.sum(shortlist[:, :30] == repeated_target[:, :30])
    cnt_40 += torch.sum(shortlist[:, :40] == repeated_target[:, :40])
    cnt_50 += torch.sum(shortlist[:, :50] == repeated_target[:, :50])
    return [cnt_10.item(), cnt_20.item(), cnt_30.item(), cnt_40.item(), cnt_50.item()]