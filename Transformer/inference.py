# Inference.py #
''' usage
python inference.py --data_root ../7SCENES \
                    --predict_root ../predict_pose \
                    --ckpt checkpoints/epoch100.pth \
                    --output_root ./refined_pose
'''

import argparse, torch, numpy as np, os
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import SevenScenesRefine
from model   import RGBDRefineTransformer
from utility import se3_exp

def geodesic(R):
    return torch.acos(torch.clamp(
        (torch.diagonal(R,0,-2,-1).sum(-1)-1)/2, -1+1e-6, 1-1e-6))

@torch.no_grad()
def inference(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = SevenScenesRefine(args.data_root, args.predict_root, path = path,train=False)
    loader   = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=10)
    print(f"Test samples: {len(test_set)}")

    model = RGBDRefineTransformer().to(device)
    ckpt  = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
    model.eval()

    for batch in tqdm(loader):
        rgb0, d0, rgbi, di = [batch[k].to(device) for k in ('rgb0','d0','rgbi','di')]
        posei = batch['pi'].to(device)
        fid   = batch['fid'].to(device)
        fid_emb = torch.sin(torch.arange(0,384, device=device)[None,:] * fid[:,None] / 1000)

        T_ref = se3_exp(model(rgb0,d0,rgbi,di,fid_emb)) @ posei   # (1,4,4)

        # output path
        src_paths = batch['path']
        for src_path in src_paths:
            src_path = src_path.strip()
            parts = Path(src_path).parts            # .../<scene>/<split>/<seq>/frame-xxxxx.color.png
            scene, split, seq, file_name = parts[-4], parts[-3], parts[-2], parts[-1]
            frame_id = file_name.split('.')[0]      # frame-000123

            out_dir = Path(args.output_root) / scene / split / seq 
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_dir / f"{frame_id}.pose.txt", T_ref[0].cpu().numpy(), fmt='%16.7e')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',    required=True)
    ap.add_argument('--predict_root', required=True)
    ap.add_argument('--ckpt',         required=True)
    ap.add_argument('--output_root',  required=True)
    ap.add_argument('--device',       default='cuda:0')
    args = ap.parse_args()
    inference(args, path="train")
    inference(args, path="test")
    
