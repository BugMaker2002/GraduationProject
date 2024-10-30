import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft

class CalculateNormPSD(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[1])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[:,use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


class MultiViewTripletLoss(nn.Module):
    def __init__(self, Fs, high_pass, low_pass, mvtl_distance):
        super(MultiViewTripletLoss, self).__init__()
        self.norm_psd = None
        if 'PSD' in mvtl_distance:
            self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        if 'MSE' in mvtl_distance:
            self.distance_func = nn.MSELoss(reduction = 'none')
        elif 'L1' in mvtl_distance:
            self.distance_func = nn.L1Loss(reduction = 'none')
        else:
            raise Exception(f"ERROR: Unknown distance metric {mvtl_distance}")

    def compare_view_lists(self, list_a, list_b):
        total_distance = 0.
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                total_distance += self.distance_func(list_a[i], list_b[j])
        return total_distance

    def forward(self, branches):   
        # Calculate NormPSD for each branch, if needed
        num_temp_views = len(branches['anc'])
        if self.norm_psd is not None:
            for key in branches.keys():
                for temp_i in range(num_temp_views):
                    # print(branches[key][temp_i].shape)
                    branches[key][temp_i] = self.norm_psd(branches[key][temp_i])
                    # print(branches[key][temp_i].shape)

        # Tally the triplet loss
        pos_loss = self.compare_view_lists(branches['anc'], branches['pos'])
        neg_loss = self.compare_view_lists(branches['anc'], branches['neg'])
        return (pos_loss - neg_loss) / num_temp_views * num_temp_views