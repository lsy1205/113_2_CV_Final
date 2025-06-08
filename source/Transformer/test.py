import os 
import argparse
import numpy as np
import torch

def main(data_dir, predict_dir, train=True):
    samples = []
    for scene in sorted(os.listdir(data_dir)):
        print(f"Processing scene: {scene}")

        if train:
            scene_path = os.path.join(data_dir, scene, "train")
        else:
            scene_path = os.path.join(data_dir, scene, "test")
        for seq in sorted(os.listdir(scene_path)):
            seq_path = os.path.join(scene_path, seq)
            print(f"  Processing sequence: {seq_path}")
            unique_frames = set()
            for frame in sorted(os.listdir(seq_path)):
                base_name = frame.split('.')[0]
                unique_frames.add(base_name)
            for frame_name in sorted(unique_frames):
                if frame_name == 'frame-000000' or not frame_name.startswith("frame-"):
                    continue
                else: 
                    data_path = os.path.join(seq_path, frame_name)

                    if train:
                        pred_path = os.path.join(predict_dir, scene, "train", seq, frame_name)
                    else:
                        pred_path = os.path.join(predict_dir, scene, "test", seq, frame_name)

                    if train:
                        samples.append(
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
                        samples.append(
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
                    print(samples[-1])
                    # input()

if __name__ == '__main__':
    data_dir = "../7SCENES"
    predict_dir = "../predict_pose"

    main(data_dir = data_dir, predict_dir = predict_dir, train=False)