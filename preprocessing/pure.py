import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import glob
import os
import yaml
from easydict import EasyDict
from tqdm import tqdm
import json

# 从 YAML 文件加载配置信息
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

class Process_PURE():
    def __init__(self, data_path=config.PURE.data_path,
                 output_path=config.PURE.output_path,
                 device=torch.device('cuda')):
        self.data_path = data_path
        self.output_path = output_path
        self.dataset_name = "PURE"
        self.device = device  
        self.dirs = self.get_raw_data()
    
    def get_raw_data(self):
        data_dirs = glob.glob(self.data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1]
            subject = int(subject_trail_val[0:2])
            dirs.append({"filename": subject_trail_val, "path": data_dir, "subject": subject})
        return dirs
    
    def save(self):
        for i in tqdm(range(len(self.dirs))):
            filename = self.dirs[i]["filename"]
            path = self.dirs[i]["path"]
            video_file = os.path.join(path, filename)
            bvp_file = os.path.join(path, f"{filename}.json")
            frames, raw_gts = self.face_detection(video_file), self.read_wave(bvp_file)
            gts = self.resample_ppg(raw_gts, frames.shape[0])
            result = {"frames": frames, "gts": gts}
            # print(frames.shape, gts.shape)
            np.save(os.path.join(self.output_path, f"{filename}.npy"), result)
    
    def face_detection(self, video_file):
        device = self.device
        mtcnn = MTCNN(device=device)
        video_list = self.read_video(video_file)
        face_list = []
        for t, frame in enumerate(video_list):
            # print(f"frame {t}")
            if t == 0:
                boxes, _, = mtcnn.detect(
                    frame)  # we only detect face bbox in the first frame, keep it in the following frames.
            if t == 0:
                box_len = np.max([boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]])
                box_half_len = np.round(box_len / 2 * 1.1).astype('int')
            box_mid_y = np.round((boxes[0, 3] + boxes[0, 1]) / 2).astype('int')
            box_mid_x = np.round((boxes[0, 2] + boxes[0, 0]) / 2).astype('int')
            cropped_face = frame[box_mid_y - box_half_len:box_mid_y + box_half_len,
                           box_mid_x - box_half_len:box_mid_x + box_half_len]
            cropped_face = cv2.resize(cropped_face, (128, 128))
            face_list.append(cropped_face)

        face_list = np.array(face_list)  # (T, H, W, C)
        return face_list
    
    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '/*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
    
    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)
    
if __name__ == '__main__':
    p = Process_PURE()
    p.save()