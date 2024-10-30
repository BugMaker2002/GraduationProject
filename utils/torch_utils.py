import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch
import torch.cuda as cutorch

class CalculateMultiView(nn.Module):
    def __init__(self, sub_length, num_views):
        super().__init__()
        self.num_views = num_views
        self.sub_length = sub_length
    def forward(self, input, zero_pad=0):
        # Pad input to be at least sub_length long
        if input.shape[-1] < self.sub_length:
            input = F.pad(input, (0, self.sub_length - input.shape[-1]))
        # Stack all random views
        views = []
        for i in range(self.num_views):
            # Random subset
            offset = torch.randint(0, input.shape[-1] - self.sub_length + 1, (1,), device=input.device)
            x = input[..., offset:offset + self.sub_length]
            views.append(x)
        return views
    
def cal_mae_rmse_r(hr_pred, hr_gt):
    mae = np.mean(np.abs(hr_pred - hr_gt))
    mse = np.mean((hr_pred - hr_gt) ** 2)
    rmse = np.sqrt(mse)
    r = np.corrcoef(hr_pred, hr_gt)[0, 1]
    return mae, rmse, r

def cal_mae_rmse_r2(hr_pred, hr_gt):
    mae = np.mean(np.abs(hr_pred - hr_gt))
    mse = np.mean((hr_pred - hr_gt) ** 2)
    rmse = np.sqrt(mse)
    r = np.corrcoef(hr_pred, hr_gt)[0, 1]
    return mae, rmse, r