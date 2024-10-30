import os
import numpy as np
from torch.utils.data.dataset import Dataset
from easydict import EasyDict
import yaml

# 从 YAML 文件加载配置信息
with open('./preprocessing/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

class UBFCDataset(Dataset):
    def __init__(self, data_path=config.UBFC_rPPG.output_path, train=True, vid_frame=300):	
        self.data_path = data_path
        self.train = train
        self.vid_frame = vid_frame
        self.train_list = []
        self.val_list = []
        val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]
        for subject in range(1,50):
            if os.path.isfile(self.data_path+'/subject%d.npy'%(subject)):
                if subject in val_subject:
                    self.val_list.append(self.data_path+'/subject%d.npy'%(subject))
                else:
                    self.train_list.append(self.data_path+'/subject%d.npy'%(subject))
                    
    def __len__(self):
        if self.train==True:
            return len(self.train_list)
        else:
            return len(self.val_list)

    def __getitem__(self, idx):
        if self.train==True:
            subject = np.load(self.train_list[idx], allow_pickle=True)
            frames, fps = subject.item().get('frames')
            gts = subject.item().get('gts')
            img_length = frames.shape[0]
            idx_start = np.random.choice(img_length-self.vid_frame)
            idx_end = idx_start+self.vid_frame
            frames, gts = frames[idx_start:idx_end], gts[idx_start:idx_end]
            frames = np.transpose(frames, (3, 0, 1, 2)).astype('float32')
        else:
            subject = np.load(self.val_list[idx], allow_pickle=True)
            frames, fps = subject.item().get('frames')
            gts = subject.item().get('gts')
            frames = frames.astype('float32')
        return frames, gts
    
class PUREDataset(Dataset):
    def __init__(self, data_path=config.PURE.output_path, train=True, vid_frame=300):	
        self.data_path = data_path
        self.train = train
        self.vid_frame = vid_frame
        self.train_list = []
        self.val_list = []
        val_file = ["10-01", "10-02", "10-03", "10-04", "10-05", "10-06",
                   "09-01", "09-02", "09-03", "09-04", "09-05", "09-06",
                   "08-01", "08-02", "08-03", "08-04", "08-05", "08-06",]
        for filename in os.listdir(self.data_path):
            if filename[:-4] in val_file:
                self.val_list.append(os.path.join(self.data_path, filename))
            else:
                self.train_list.append(os.path.join(self.data_path, filename))
                    
    def __len__(self):
        if self.train==True:
            return len(self.train_list)
        else:
            return len(self.val_list)

    def __getitem__(self, idx):
        if self.train==True:
            subject = np.load(self.train_list[idx], allow_pickle=True)
            frames = subject.item().get('frames')
            gts = subject.item().get('gts')
            
            img_length = frames.shape[0]
            # print(img_length)
            idx_start = np.random.choice(img_length-self.vid_frame)
            idx_end = idx_start+self.vid_frame
            frames, gts = frames[idx_start:idx_end], gts[idx_start:idx_end]
            frames = np.transpose(frames, (3, 0, 1, 2)).astype('float32')
        else:
            subject = np.load(self.val_list[idx], allow_pickle=True)
            frames = subject.item().get('frames')
            gts = subject.item().get('gts')
            frames = frames.astype('float32')
        return frames, gts
    
