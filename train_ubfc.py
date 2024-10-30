import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import torch
from model.PhysNetModel import PhysNet
from loss.loss import ContrastLoss
from loss.IrrelevantPowerRatio import IrrelevantPowerRatio
from dataset import *
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import random
from model.UniFormer import Uniformer
from model.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from model.FrequencyContrast import * 
from model.PhysNetUpsample import PhysNetUpsample
from utils.utils_sig import *
from loss.MultiViewTripletLoss import MultiViewTripletLoss
from utils.torch_utils import *
from torch.optim.lr_scheduler import StepLR

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

total_epoch = 30  # total number of epochs for training the model
lr = 1e-6  # learning rate
in_ch = 3  # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.
fs = 30  # video frame rate, TODO: modify it if your video frame rate is not 30 fps. 

# 这两个值，physnet对应的是T=300,S=2;physformer对应的是T=160,S=4
T = 160  # frame

result_dir = './results/ubfc/physformer'  # store checkpoints and training recording

# define the dataloader
train_dataset = UBFCDataset(train=True, vid_frame=T)  # please read the code about H5Dataset when preparing your dataset
train_dataloader = DataLoader(train_dataset, batch_size=4,  # two videos for contrastive learning
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)  # TODO: If you run the code on Windows, please remove num_workers=4.

val_dataset = UBFCDataset(train=False, vid_frame=T)  # please read the code about H5Dataset when preparing your dataset
val_dataloader = DataLoader(val_dataset, batch_size=1,  # two videos for contrastive learning
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)  # TODO: If you run the code on Windows, please remove num_workers=4.

# define the model and loss
# model = PhysNet(S, in_ch=in_ch).to(device).train()
# model = ViT_ST_ST_Compact3_TDC_gra_sharp(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12,
#                                           patches=(4, 4, 4), theta=0.7,
#                                          dropout_rate=0.1).to(device).train()
# model = Uniformer(
#         num_classes = 1000,                 # number of output classes
#         dims = (64, 64, 64, 64),
#         depths = (2, 2, 2, 2),
#         mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
#     ).to(device).train()
model = PPGExtractor(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
                     patches=(4, 4, 4), theta=0.7, dropout_rate=0.1).to(device).train()
# model = PhysNetUpsample().to(device).train()

get_temp_views = CalculateMultiView(150, 4)
loss_func = MultiViewTripletLoss(30, 40, 250, ['PSD', 'MSE'])
IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)
opt = optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)


@torch.no_grad()
def dl_model(imgs_clip):
    # model inference
    img_batch = imgs_clip
    # print(img_batch.shape)
    img_batch = img_batch.permute((3,0,1,2))
    # 在img_batch前面新增了一个批量大小的维度（批量大小为1）
    img_batch = img_batch.unsqueeze(0).to(device)

    rppg = model(img_batch)[:,-1, :] # (B, 1, T) -> (B, T)
    rppg = rppg[0].detach().cpu().numpy() # 因为验证的时候是每次取出一个样本，所以取rppg[0]就行了
    return rppg
    
@torch.no_grad()
def validate(val_loader, clip_len):
    hr_pred = []
    hr_gt = []
    for frames, gts in val_loader:
        # print(frames.shape, gts.shape)
        frames, gts = frames.squeeze(0), gts.squeeze(0)
        # print(frames.shape, gts.shape)
        img_length = frames.shape[0]
        # print(img_length)
        num_blocks = img_length // clip_len
        # print(num_blocks)

        hr_per_video = []
        gt_per_video = []
        for i in range(num_blocks):
            # print(frames.shape)
            img_clip = frames[i * clip_len:(i + 1) * clip_len]
            # print(img_clip.shape)
            rppg_clip = dl_model(img_clip)
            gt_clip = gts[i * clip_len:(i + 1) * clip_len]
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
    mae, rmse, r = cal_mae_rmse_r(hr_pred, hr_gt)
    return mae, rmse, r

total_train_loss = []
total_val_loss = []
for e in range(total_epoch):
    # model.train()
    print(f"train start in epoch {e}")
    train_loss = []
    for it in range(np.round(60 / (T / fs)).astype(
            'int')):  # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
        for imgs, gts in train_dataloader:  # dataloader randomly samples a video clip with length T
            x_anchor = imgs.to(device)
            T = x_anchor.shape[2] # 获取帧数
            freq_factor = 1.25 + (torch.rand(1, device=x_anchor.device) / 4) 
            # 生成一个值在1.25~1.5之间的随机数
            target_size = int(T / freq_factor)
            resampler = nn.Upsample(size=(target_size, x_anchor.shape[3], x_anchor.shape[4]),
                                mode='trilinear',
                                align_corners=False)
            x_neg = resampler(x_anchor) # 生成负样本 torch.Size([2, 3, 117, 128, 128])
            # print("x_neg: ", x_neg.shape)
            x_neg = F.pad(x_neg, (0, 0, 0, 0, 0, T - target_size)) # torch.Size([2, 3, 160, 128, 128])
            # print("x_neg after padding: ", x_neg.shape)
            y_a = model(x_anchor) # anchor对应的rppg信号 # torch.Size([2, 1, 160])
            # print("y_a: ", y_a.shape)
            y_n = model(x_neg) # 负样本对应的rppg信号 # torch.Size([2, 1, 160])
            # print("y_n: ", y_n.shape)
            y_n = y_n[:, :, :target_size] # 移除padding # torch.Size([2, 1, 117])
            # print("y_n remove padding: ", y_n.shape)
            resampler2 = nn.Upsample(size=(T,), mode='linear', align_corners=False) # 将负样本重新变换成原来，生成正样本
            y_p = resampler2(y_n)
            # print(y_a.shape, y_n.shape, y_p.shape) # torch.Size([2, 1, 160]) torch.Size([2, 1, 117]) torch.Size([2, 1, 160])
            branches = {}
            branches['anc'] = y_a.squeeze(1)
            branches['neg'] = y_n.squeeze(1)
            branches['pos'] = y_p.squeeze(1)
            # Sample random views for each branch
            for key, branch in branches.items():
                branches[key] = get_temp_views(branch)
            # define the loss functions
            loss = loss_func(branches)
            # print(f"loss: {loss.sum():.2f}")
            train_loss.append(loss.sum())
            # optimize
            opt.zero_grad()
            loss.sum().backward()
            opt.step()

            # evaluate irrelevant power ratio during training
            ipr = torch.mean(IPR(y_a.squeeze(1).clone().detach()))
    # scheduler.step(sum(train_loss) / len(train_loss))
    # scheduler.step()
    # model.eval()
    mae, rmse, r = validate(val_dataloader, 160)
    total_train_loss.append(sum(train_loss) / len(train_loss))
    # print(f"epoch{e} train loss: {sum(train_loss) / len(train_loss):.2f}")
    total_val_loss.append(mae)
    print(f"epoch{e} mae: {mae:.2f}, rmse: {rmse:.2f}, r: {r:.2f}")
    # save model checkpoints
    torch.save(model.state_dict(), result_dir + '/epoch%d.pt' % e)
    print(f"model saved in epoch {e}")

# epochs = range(total_epoch)
# # 创建第一个图表，用于绘制训练损失
# plt.figure(figsize=(8, 4))  # 设置图表尺寸
# plt.subplot(1, 2, 1)  # 子图布局：1行2列，第1个子图
# plt.plot(epochs, total_train_loss, 'bo-')
# plt.title('Training Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)

# # 创建第二个图表，用于绘制验证损失
# plt.subplot(1, 2, 2)  # 子图布局：1行2列，第2个子图
# plt.plot(epochs, total_val_loss, 'ro-')
# plt.title('Validation Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# # 调整子图布局，避免重叠
# plt.tight_layout()
# # 将图表保存为图片文件
# plt.savefig('training_and_validation_loss_plots.png')
# # 显示图表
# plt.show()