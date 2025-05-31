import os
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

class SevenScenesRefine(Dataset):
    def __init__(self, data_dir, predict_dir, path="test", train=True):
        self.samples = []
        self.train = train

        for scene in sorted(os.listdir(data_dir)):
            if self.train:
                scene_path = os.path.join(data_dir, scene, "train")
            else:
                scene_path = os.path.join(data_dir, scene, path)
            for seq in sorted(os.listdir(scene_path)):
                seq_path = os.path.join(scene_path, seq)
                # print(f"Processing {scene_path}...")
                unique_frames = set()
                for frame in sorted(os.listdir(seq_path)):
                    base_name = frame.split('.')[0]
                    unique_frames.add(base_name)
                for frame_name in sorted(unique_frames):
                    if frame_name == 'frame-000000' or not frame_name.startswith("frame-"):
                        continue
                    else: 
                        data_path = os.path.join(seq_path, frame_name)

                        if self.train:
                            pred_path = os.path.join(predict_dir, scene, "train", seq, frame_name)
                        else:
                            pred_path = os.path.join(predict_dir, scene, path, seq, frame_name)
                        if self.train:
                            self.samples.append(
                                dict(
                                    rgb0 = seq_path + '/frame-000000.color.png',
                                    d0   = seq_path + '/frame-000000.depth.proj.png',
                                    p0   = seq_path + '/frame-000000.pose.txt',
                                    rgbi = data_path + '.color.png',
                                    di   = data_path + '.depth.proj.png',
                                    pi   = pred_path + '.pose.txt',
                                    pgti = data_path + '.pose.txt',
                                    fx   = 525,
                                    fy   = 525,
                                    cx   = 320,
                                    cy   = 240, 
                                    fid  = int(frame_name.split('-')[1])
                                )
                            )
                        else:
                            self.samples.append(
                                dict(
                                    rgb0 = seq_path + '/frame-000000.color.png',
                                    d0   = seq_path + '/frame-000000.depth.proj.png',
                                    p0   = seq_path + '/frame-000000.pose.txt',
                                    rgbi = data_path + '.color.png',
                                    di   = data_path + '.depth.proj.png',
                                    pi   = pred_path + '.pose.txt',
                                    fx   = 525,
                                    fy   = 525,
                                    cx   = 320,
                                    cy   = 240, 
                                    fid  = int(frame_name.split('-')[1])
                                )
                            )
                            


    def _load_rgb(self, path):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,384), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2,0,1).float()/255.
        return (img - MEAN)/STD            # (3,384,512)

    def _load_depth(self, path):
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        d = cv2.resize(d, (512, 384), interpolation=cv2.INTER_NEAREST)

        d = d.astype(np.float32)
        d = d / 1000.0

        d = torch.from_numpy(d).unsqueeze(0).float()  # (1, H, W)
        return d

    def _load_pose(self, path):
        return torch.from_numpy(np.loadtxt(path, dtype=np.float32))  # (4,4)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.train:
            sample = dict(
                rgb0 = self._load_rgb(s['rgb0']),
                d0   = self._load_depth(s['d0']),
                p0   = self._load_pose(s['p0']),
                rgbi = self._load_rgb(s['rgbi']),
                di   = self._load_depth(s['di']),
                pi   = self._load_pose(s['pi']),
                pgti = self._load_pose(s['pgti']),
                fx   = torch.tensor(s['fx'], dtype=torch.float32),
                fy   = torch.tensor(s['fy'], dtype=torch.float32),
                cx   = torch.tensor(s['cx'], dtype=torch.float32),
                cy   = torch.tensor(s['cy'], dtype=torch.float32),
                fid  = torch.tensor(s['fid'], dtype=torch.long),
            )
        else:
            sample = dict(
                rgb0 = self._load_rgb(s['rgb0']),
                d0   = self._load_depth(s['d0']),
                p0   = self._load_pose(s['p0']),
                rgbi = self._load_rgb(s['rgbi']),
                di   = self._load_depth(s['di']),
                pi   = self._load_pose(s['pi']),
                fx   = torch.tensor(s['fx'], dtype=torch.float32),
                fy   = torch.tensor(s['fy'], dtype=torch.float32),
                cx   = torch.tensor(s['cx'], dtype=torch.float32),
                cy   = torch.tensor(s['cy'], dtype=torch.float32),
                fid  = torch.tensor(s['fid'], dtype=torch.long),
                path = s['rgbi']
            )
        return sample

    def __len__(self): return len(self.samples)
