import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import re
import glob
import os
import yaml
from easydict import EasyDict
from tqdm import tqdm

# 从 YAML 文件加载配置信息
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

class Process_UBFCrPPG():
    def __init__(self, data_path=config.UBFC_rPPG.data_path,
                 output_path=config.UBFC_rPPG.output_path,
                 device=torch.device('cuda')):
        self.data_path = data_path
        self.output_path = output_path
        self.dataset_name = "UBFC_rPPG"
        self.device = device
        self.dirs = self.get_raw_data()

    def get_raw_data(self):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(self.data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def save(self):
        for i in tqdm(range(len(self.dirs))):
            index = self.dirs[i]["index"]
            subject_path = self.dirs[i]["path"]
            video_file = os.path.join(subject_path, "vid.avi")
            bvp_file = os.path.join(subject_path, "ground_truth.txt")
            frames, gts = self.face_detection(video_file), self.read_wave(bvp_file)
            result = {"frames": frames, "gts": gts}
            np.save(os.path.join(self.output_path, f"{index}.npy"), result)


    def face_detection(self, video_file):
        device = self.device
        mtcnn = MTCNN(device=device)
        video_list, fps = self.read_video(video_file)
        face_list = []
        for t, frame in enumerate(video_list):
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
        return face_list, fps

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        fps = VidObj.get(cv2.CAP_PROP_FPS)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return frames, fps

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

if __name__ == '__main__':
    p = Process_UBFCrPPG()
    p.save()
