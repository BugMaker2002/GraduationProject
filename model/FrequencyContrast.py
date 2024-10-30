import os
import sys
import numpy as np
from typing import Optional
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import math
# sys.path.append("../utils")
# from torch_utils import *
# sys.path.append("../loss")
# from MultiViewTripletLoss import *

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
# 形状只会被self.conv层所改变
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

# x = torch.rand(size=(3, 4, 6))
# shape = (3, -1, 2)
# print(split_last(x, shape).shape) # torch.Size([3, 4, 3, 1, 2])
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

# x = torch.rand(size=(3, 4, 6, 7))
# n_dims = 2
# print(merge_last(x, n_dims).shape) # torch.Size([3, 4, 42])
def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores




class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        # h = self.proj(Atten)
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        # h = self.pwff(self.norm2(x))
        x = x + h
        return x, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class PPGExtractor(nn.Module):

    def __init__(
        self, 
        gra_sharp=2.0,
        S: int = 4,
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2
    ):
        super().__init__()

        self.S = S
        self.gra_sharp = gra_sharp
        self.frame = frame  
        self.dim = dim              
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
           
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x):

        # b is batch number, c channels, t frame, fh frame height, and fw frame width
        b, c, t, fh, fw = x.shape
        # print("x.shape: ", b, c, t, fh, fw)
        
        x = self.Stem0(x)
        # print("Stem0: ", x.shape) # B,dim//4,T,H/2,W/2
        x = self.Stem1(x)
        # print("Stem1: ", x.shape) # B,dim//2,T,H/4,W/4
        x = self.Stem2(x)  
        # print("Stem2: ", x.shape) # B,dim,T,H/8,W/8
        
        x = self.patch_embedding(x)  
        # print("patch_embedding: ", x.shape) # B,dim,T/patch,H/8/patch,W/8/patch
        x = x.flatten(2).transpose(1, 2)  
        # print("x.flatten(2).transpose(1, 2): ", x.shape) # B,seq_len,dim
        
        
        Trans_features, Score1 =  self.transformer1(x, self.gra_sharp)  
        # print("Trans_features, Score1: ", Trans_features.shape, Score1.shape) # B,seq_len,dim
        
        Trans_features2, Score2 =  self.transformer2(Trans_features, self.gra_sharp)  
        # print("Trans_features2, Score2: ", Trans_features2.shape, Score2.shape) # B,seq_len,dim
        
        Trans_features3, Score3 =  self.transformer3(Trans_features2, self.gra_sharp)  
        # print("Trans_features3, Score3: ", Trans_features3.shape, Score3.shape) # B,seq_len,dim
        
        # upsampling heads
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4) 
        # print("features_last: ", features_last.shape) # B,dim,T/patch,H/8/patch,W/8/patch
        
        features_last = self.upsample(features_last)		   
        # print("upsample: ", features_last.shape) # B,dim,T/patch*2,H/8/patch,W/8/patch
        features_last = self.upsample2(features_last)		    
        # print("upsample2: ", features_last.shape) # B,dim//2,T/patch*4,H/8/patch,W/8/patch
        
        features_last = torch.mean(features_last,3)     
        features_last = torch.mean(features_last,3)     
        rPPG = self.ConvBlockLast(features_last)    
        
        return rPPG
    
# class FreqContra(nn.Module):
#     def __init__(self, gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
#                                              patches=(4, 4, 4), theta=0.7, dropout_rate=0.1, 
#                                              mvtl_window_size=150, mvtl_number_views=4):
#         super().__init__()
#         self.backbone = PPGExtractor(gra_sharp=gra_sharp, S=S, dim=dim, ff_dim=ff_dim, num_heads=num_heads, 
#                                      num_layers=num_layers, patches=patches, theta=theta, dropout_rate=dropout_rate)
#         self.get_temp_views = CalculateMultiView(mvtl_window_size, mvtl_number_views)
    
#     def forward(self, x_anchor):
#         T = x_anchor.shape[2] # 获取帧数
#         freq_factor = 1.25 + (torch.rand(1) / 4) 
#         # 生成一个值在1.25~1.5之间的随机数
#         target_size = int(T / freq_factor)
#         resampler = nn.Upsample(size=(target_size, x_anchor.shape[3], x_anchor.shape[4]),
#                                 mode='trilinear',
#                                 align_corners=False)
#         x_neg = resampler(x_anchor) # 生成负样本 torch.Size([2, 3, 117, 128, 128])
#         # print("x_neg: ", x_neg.shape)
#         x_neg = F.pad(x_neg, (0, 0, 0, 0, 0, T - target_size)) # torch.Size([2, 3, 160, 128, 128])
#         # print("x_neg after padding: ", x_neg.shape)
#         y_a = self.backbone(x_anchor) # anchor对应的rppg信号 # torch.Size([2, 1, 160])
#         # print("y_a: ", y_a.shape)
#         y_n = self.backbone(x_neg) # 负样本对应的rppg信号 # torch.Size([2, 1, 160])
#         # print("y_n: ", y_n.shape)
#         y_n = y_n[:, :, :target_size] # 移除padding # torch.Size([2, 1, 117])
#         # print("y_n remove padding: ", y_n.shape)
#         resampler2 = nn.Upsample(size=(T,), mode='linear', align_corners=False) # 将负样本重新变换成原来，生成正样本
#         y_p = resampler2(y_n)
#         # print(y_a.shape, y_n.shape, y_p.shape) # torch.Size([2, 1, 160]) torch.Size([2, 1, 117]) torch.Size([2, 1, 160])
#         branches = {}
#         branches['anc'] = y_a.squeeze(1)
#         branches['neg'] = y_n.squeeze(1)
#         branches['pos'] = y_p.squeeze(1)
#         # Sample random views for each branch
#         for key, branch in branches.items():
#             branches[key] = self.get_temp_views(branch)
        
#         return branches
        
    
if __name__ == '__main__':
    # # 模型， rppg提取器，从一个视频当中提取rppg,shape为(B, 1, T)
    model = PPGExtractor(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
                                             patches=(4, 4, 4), theta=0.7, dropout_rate=0.1)
    # # 输入视频
    # x_anchor = torch.rand(size=(2, 3, 160, 128, 128))
    
    # # 改变x的频率，生成对应的负样本
    # T = x_anchor.shape[2] # 获取帧数
    # freq_factor = 1.25 + (torch.rand(1) / 4) # 生成一个值在1.25~1.5之间的随机数
    # target_size = int(T / freq_factor)
    # resampler = nn.Upsample(size=(target_size, x_anchor.shape[3], x_anchor.shape[4]),
    #                             mode='trilinear',
    #                             align_corners=False)
    
    # x_neg = resampler(x_anchor) # 生成负样本 torch.Size([2, 3, 117, 128, 128])
    # # print("x_neg: ", x_neg.shape)
    # x_neg = F.pad(x_neg, (0, 0, 0, 0, 0, T - target_size)) # torch.Size([2, 3, 160, 128, 128])
    # # print("x_neg after padding: ", x_neg.shape)
    
    # y_a = model(x_anchor) # anchor对应的rppg信号 # torch.Size([2, 1, 160])
    # # print("y_a: ", y_a.shape)
    # y_n = model(x_neg) # 负样本对应的rppg信号 # torch.Size([2, 1, 160])
    # # print("y_n: ", y_n.shape)
    # y_n = y_n[:, :, :target_size] # 移除padding # torch.Size([2, 1, 117])
    # # print("y_n remove padding: ", y_n.shape)
    # resampler2 = nn.Upsample(size=(T,), mode='linear', align_corners=False) # 将负样本重新变换成原来，生成正样本
    # y_p = resampler2(y_n)
    # # print(y_a.shape, y_n.shape, y_p.shape) # torch.Size([2, 1, 160]) torch.Size([2, 1, 117]) torch.Size([2, 1, 160])
    # fc = FreqContra()
    # x_anchor = torch.rand(size=(2, 3, 160, 128, 128))
    # branches = fc(x_anchor)
    # # for name, param in fc.named_parameters():
    # #     print(name)
    # print(len(branches['anc']), branches['anc'][0].shape)
    # print(len(branches['pos']), branches['pos'][0].shape)
    # print(len(branches['neg']), branches['neg'][0].shape)
    # loss = MultiViewTripletLoss(30, 40, 250, ['PSD', 'MSE'])
    # print(loss(branches).shape)
    
    # model = PPGExtractor(gra_sharp=2.0, S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
    #                                          patches=(4, 4, 4), theta=0.7, dropout_rate=0.1)
    # fc = FreqContra()
    # for param1, param2 in zip(model.named_parameters(), fc.named_parameters()):
    #     print(param1[0] == param2[0][9:])
