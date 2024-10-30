import numpy as np
import h5py
from torch import nn
import torch
from model.PhysNetModel import PhysNet
from dataset import *
import json
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from utils.utils_sig import *
from model.FrequencyContrast import * 
from utils.torch_utils import *

e = 29  # the model checkpoint at epoch e
train_exp_dir = './results/pure/physformer'  # training experiment directory
T = 160
in_ch = 3

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

else:
    device = torch.device('cpu')

# model = PhysNet(S, in_ch=in_ch).to(device).eval()
# model = ViT_ST_ST_Compact3_TDC_gra_sharp(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12,
#                                          patches=(4, 4, 4), theta=0.7,
#                                          dropout_rate=0.1).to(device).eval()
model = PPGExtractor(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
                     patches=(4, 4, 4), theta=0.7, dropout_rate=0.1).to(device).eval()

model.load_state_dict(torch.load(train_exp_dir + '/epoch%d.pt' % (e), map_location=device))  # load weights to the model
# model.load_state_dict(torch.load(train_exp_dir + '/best_epoch.pt', map_location=device))  # load weights to the model

dataset = PUREDataset(train=False, vid_frame=T)  # please read the code about H5Dataset when preparing your dataset
dataloader = DataLoader(dataset, batch_size=1,  # two videos for contrastive learning
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)  # TODO: If you run the code on Windows, please remove num_workers=4.

@torch.no_grad()
def dl_model(imgs_clip):
    # model inference
    img_batch = imgs_clip
    # print(img_batch.shape)
    img_batch = img_batch.permute((3,0,1,2))
    # 在img_batch前面新增了一个批量大小的维度（批量大小为1）
    img_batch = img_batch.unsqueeze(0).to(device)

    rppg = model(img_batch)[:,-1, :] # (1, 5, T) -> (1, T)
    rppg = rppg[0].detach().cpu().numpy()
    return rppg

hr_pred = []
hr_gt = []
for imgs, gts in dataloader:
    imgs = imgs.squeeze(0)
    gts = gts.squeeze(0)
    # print(imgs.shape, gts.shape)
    
    img_len = imgs.shape[0]
    num_blocks = img_len // T
    
    hr_per_video = []
    gt_per_video = []
    for i in range(num_blocks):
        img_clip = imgs[i*T:(i+1)*T]
        rppg_clip = dl_model(img_clip)
        gt_clip = gts[i*T:(i+1)*T]
        # print(img_clip.shape, gt_clip.shape)
        
        # 标准化信号
        rppg_norm = normalize(rppg_clip)
        bvp_norm = normalize(gt_clip)

        fs = 30  # 假设采样率为 30Hz

        # 对标准化信号进行滤波
        rppg_filtered = butter_bandpass(rppg_norm, lowcut=0.6, highcut=4, fs=fs)
        bvp_filtered = butter_bandpass(bvp_norm, lowcut=0.6, highcut=4, fs=fs)

        # 计算预测心率和地面真值心率
        hr_rppg, rppg_psd, rppg_hr = hr_fft(rppg_filtered, fs=fs)
        hr_bvp, bvp_psd, bvp_hr = hr_fft(bvp_filtered, fs=fs)
        
        hr_per_video.append(hr_rppg)
        gt_per_video.append(hr_bvp)
    
    hr_pred.append(np.mean(hr_per_video))
    hr_gt.append(np.mean(gt_per_video))
    
hr_pred = np.array(hr_pred)
hr_gt = np.array(hr_gt)
mae, rmse, r = cal_mae_rmse_r2(hr_pred, hr_gt)
# 显示评估指标和预测心率
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Pearson's Correlation Coefficient (r): {r:.2f}")