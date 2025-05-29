#!/usr/bin/env python3
"""
Convert 7-Scenes into DUSt3R-friendly .npz + pair-list.
Usage:
  python preprocess_7scenes.py \
      --raw_root  data/7scenes_raw \
      --out_root  data/7scenes_dust3r \
      --resize    640 480 \
      --near_pairs 1 \
      --far_pairs  3 20 40
"""
import argparse, json, numpy as np, cv2, imageio, tqdm, shutil
from pathlib import Path

K_DEF = np.array([[525,0,320],
                  [0,525,240],
                  [0,0,1]], np.float32)

def save_npz(rgb, depth_mm, pose, out):
    depth = depth_mm.astype(np.float32) / 1_000.   # → m
    np.savez_compressed(out,
        rgb=rgb[..., ::-1],          # BGR→RGB
        depth=depth,
        intr=K_DEF,
        pose=pose.astype(np.float32))

def frame_paths(seq_dir):
    return sorted(seq_dir.glob('frame-*.color.png'))

def make_pairs(n, near=1, far_k=3, far_min=20, far_max=40):
    pairs = []
    for i in range(n):
        # temporal鄰居
        j = i + near
        if j < n: pairs.append((i,j))
        # long-baseline
        potential_far_range_start = i + far_min
        potential_far_range_end = min(i + far_max, n)

        if potential_far_range_start < potential_far_range_end:
            selectable_indices = list(range(potential_far_range_start, potential_far_range_end))
        
        num_selectable = len(selectable_indices)

        if num_selectable > 0:
            actual_far_k_to_sample = min(far_k, num_selectable)
            
            if actual_far_k_to_sample > 0:
                rng = np.random.default_rng(i)
                far = rng.choice(selectable_indices, size=actual_far_k_to_sample, replace=False)
                pairs += [(i, int(f)) for f in far]
    return pairs

def main(args):
    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    if out_root.exists(): shutil.rmtree(out_root)
    for scene in tqdm.tqdm(list(raw_root.iterdir()), desc='scenes'):
        print(raw_root, out_root)
        print(f'Processing scene: {scene.name}')

        train_split_dir = scene/'train'
        
        for seq in train_split_dir.iterdir():
            print(f'  Processing sequence: {seq.name}')
            frames = frame_paths(seq)
            out_seq = out_root/scene.name/seq.name
            out_seq.mkdir(parents=True, exist_ok=True)
            # 1️⃣ 逐張存 .npz
            for idx, c_path in enumerate(tqdm.tqdm(frames, leave=False)):
                stem = c_path.stem.replace('.color','')
                d_path = seq/f'{stem}.depth.proj.png'
                p_path = seq/f'{stem}.pose.txt'
                rgb   = cv2.imread(str(c_path))         # BGR uint8
                if args.resize:
                    rgb = cv2.resize(rgb, tuple(args.resize))
                depth = imageio.v2.imread(d_path)       # uint16
                pose  = np.loadtxt(p_path)              # 4×4 (cam→world)
                save_npz(rgb, depth, pose, out_seq/f'{idx:06d}.npz')
            # 2️⃣ 產生 pair-list
            pairs = make_pairs(len(frames),
                               near=args.near_pairs,
                               far_k=args.far_pairs[0],
                               far_min=args.far_pairs[1],
                               far_max=args.far_pairs[2])
            np.savetxt(out_seq/'pairs.txt', np.array(pairs, int), fmt='%d')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--resize', nargs=2, type=int, default=None)
    ap.add_argument('--near_pairs', type=int, default=1)
    ap.add_argument('--far_pairs', nargs=3, type=int, default=[3,20,40])
    main(ap.parse_args())
